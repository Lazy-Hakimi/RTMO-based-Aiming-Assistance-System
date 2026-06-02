/**
 * ESP32-S3 HID Mouse Device Firmware
 * 
 * 功能: 接收来自 Jetson AGX Xavier 的串口指令，模拟USB HID鼠标设备
 * 
 * 硬件: ESP32-S3 DevKitC-1 (或其他带原生USB的ESP32-S3开发板)
 * 接线: 
 *   - ESP32-S3 USB口 → 游戏主机 (USB)
 *   - ESP32-S3 UART (RX/TX) → Jetson Xavier UART (TX/RX交叉连接)
 *     或使用USB-CDC (同一USB口虚拟串口)
 * 
 * 开发环境: Arduino IDE 2.x + ESP32 Board Package 2.0.14+
 * 配置:
 *   - Board: "ESP32S3 Dev Module"
 *   - USB Mode: "USB-OTG (TinyUSB)"
 *   - USB CDC On Boot: Enabled
 *   - Upload Mode: "USB-OTG CDC (TinyUSB)" 或 "UART0/Hardware CDC"
 * 
 * 协议: 见 src/esp32_bridge.py 协议定义
 */

#include "USB.h"
#include "USBHIDMouse.h"
#include <cstring>

// ===== 协议常量 =====
#define PROTO_VERSION     0x01

// 接收包头
#define HEADER_RX_0       0xAA
#define HEADER_RX_1       0x55
// 发送包头
#define HEADER_TX_0       0xBB
#define HEADER_TX_1       0x66

// 指令类型 (接收)
#define CMD_MOUSE_MOVE    0x01
#define CMD_MOUSE_BUTTON  0x02
#define CMD_MOUSE_WHEEL   0x03
#define CMD_MOUSE_COMBINED 0x04
#define CMD_HEARTBEAT     0x05
#define CMD_STATUS_QUERY  0x06
#define CMD_CONFIG_SET    0x07

// 响应类型 (发送)
#define RESP_STATUS       0x10
#define RESP_ERROR        0x11
#define RESP_HEARTBEAT_ACK 0x12

// 状态码
#define STATUS_OK         0x00
#define STATUS_ERROR      0x01
#define STATUS_BUSY       0x02

// 配置
#define SERIAL_BAUDRATE   921600
#define MAX_PACKET_SIZE   64
#define QUEUE_SIZE        32

// ===== 全局对象 =====
USBHIDMouse Mouse;

// 统计数据
volatile uint32_t txCount = 0;
volatile uint32_t rxCount = 0;
volatile uint32_t errorCount = 0;
volatile uint32_t lastPacketTime = 0;

// 固件版本
#define FIRMWARE_VERSION  0x01

// 鼠标状态
struct MouseState {
    int8_t dx;
    int8_t dy;
    uint8_t buttons;
    int8_t wheelV;
    int8_t wheelH;
};

// 指令队列
MouseState cmdQueue[QUEUE_SIZE];
volatile uint8_t queueHead = 0;
volatile uint8_t queueTail = 0;
volatile uint8_t queueCount = 0;

// 解析状态机
enum ParseState {
    STATE_WAIT_HEADER0,
    STATE_WAIT_HEADER1,
    STATE_VERSION,
    STATE_CMD_TYPE,
    STATE_LEN_LOW,
    STATE_LEN_HIGH,
    STATE_DATA,
    STATE_CHECKSUM
};

ParseState parseState = STATE_WAIT_HEADER0;
uint8_t rxBuffer[256];
uint16_t rxIndex = 0;
uint16_t rxDataLen = 0;
uint8_t rxCmdType = 0;
uint8_t calcChecksum = 0;

// ===== 函数声明 =====
void processPacket(uint8_t cmd, uint8_t* data, uint16_t len);
void sendResponse(uint8_t respType, uint8_t status, uint8_t* data, uint16_t len);
void enqueueMouseMove(int8_t dx, int8_t dy, uint8_t buttons, int8_t wheelV);
bool dequeueMouseMove(MouseState* state);
void flushMouseQueue();

// ===== 初始化 =====
void setup() {
    // 初始化USB串口 (用于调试和接收指令)
    Serial.begin(SERIAL_BAUDRATE);
    
    // 等待串口就绪 (调试用)
    while (!Serial && millis() < 3000) {
        delay(10);
    }
    
    // 初始化USB HID鼠标
    Mouse.begin();
    USB.begin();
    
    // 等待USB枚举完成
    delay(500);
    
    Serial.println("========================================");
    Serial.println("ESP32-S3 HID Mouse Device");
    Serial.println("Firmware v1.0");
    Serial.println("========================================");
    Serial.println("Waiting for commands...");
}

// ===== 主循环 =====
void loop() {
    // 1. 读取串口数据并解析
    while (Serial.available() > 0) {
        uint8_t byte = Serial.read();
        parseByte(byte);
    }
    
    // 2. 处理鼠标队列 (维持1000Hz等效)
    flushMouseQueue();
    
    // 3. 维持响应速度
    delayMicroseconds(500);  // ~1000Hz 循环
}

// ===== 字节解析状态机 =====
void parseByte(uint8_t byte) {
    switch (parseState) {
        case STATE_WAIT_HEADER0:
            if (byte == HEADER_RX_0) {
                parseState = STATE_WAIT_HEADER1;
                calcChecksum = byte;
            }
            break;
            
        case STATE_WAIT_HEADER1:
            if (byte == HEADER_RX_1) {
                parseState = STATE_VERSION;
                calcChecksum += byte;
            } else {
                parseState = (byte == HEADER_RX_0) ? STATE_WAIT_HEADER1 : STATE_WAIT_HEADER0;
            }
            break;
            
        case STATE_VERSION:
            if (byte == PROTO_VERSION) {
                parseState = STATE_CMD_TYPE;
                calcChecksum += byte;
            } else {
                parseState = STATE_WAIT_HEADER0;
                errorCount++;
            }
            break;
            
        case STATE_CMD_TYPE:
            rxCmdType = byte;
            parseState = STATE_LEN_LOW;
            calcChecksum += byte;
            break;
            
        case STATE_LEN_LOW:
            rxDataLen = byte;
            parseState = STATE_LEN_HIGH;
            calcChecksum += byte;
            break;
            
        case STATE_LEN_HIGH:
            rxDataLen |= (byte << 8);
            rxIndex = 0;
            parseState = STATE_DATA;
            calcChecksum += byte;
            if (rxDataLen > 256) {
                // 数据长度异常
                parseState = STATE_WAIT_HEADER0;
                errorCount++;
            }
            break;
            
        case STATE_DATA:
            if (rxIndex < rxDataLen) {
                rxBuffer[rxIndex++] = byte;
                calcChecksum += byte;
            }
            if (rxIndex >= rxDataLen) {
                parseState = STATE_CHECKSUM;
            }
            break;
            
        case STATE_CHECKSUM:
            calcChecksum &= 0xFF;
            if (byte == calcChecksum) {
                // 校验通过，处理数据包
                rxCount++;
                lastPacketTime = millis();
                processPacket(rxCmdType, rxBuffer, rxDataLen);
            } else {
                // 校验失败
                errorCount++;
                // 发送错误响应
                uint8_t errData[2] = {0x01, calcChecksum};
                sendResponse(RESP_ERROR, STATUS_ERROR, errData, 2);
            }
            parseState = STATE_WAIT_HEADER0;
            break;
    }
}

// ===== 处理接收到的数据包 =====
void processPacket(uint8_t cmd, uint8_t* data, uint16_t len) {
    switch (cmd) {
        case CMD_MOUSE_MOVE: {
            // 数据: dx(int16), dy(int16), buttons(uint8)
            if (len >= 5) {
                int16_t dx = (int16_t)(data[0] | (data[1] << 8));
                int16_t dy = (int16_t)(data[2] | (data[3] << 8));
                uint8_t buttons = data[4];
                enqueueMouseMove((int8_t)dx, (int8_t)dy, buttons, 0);
            }
            break;
        }
        
        case CMD_MOUSE_BUTTON: {
            // 数据: button(uint8), state(uint8)
            if (len >= 2) {
                uint8_t button = data[0];
                uint8_t state = data[1];
                
                if (state == 1) {
                    // 按下
                    switch (button) {
                        case 0: Mouse.press(MOUSE_LEFT); break;
                        case 1: Mouse.press(MOUSE_RIGHT); break;
                        case 2: Mouse.press(MOUSE_MIDDLE); break;
                    }
                } else {
                    // 释放
                    switch (button) {
                        case 0: Mouse.release(MOUSE_LEFT); break;
                        case 1: Mouse.release(MOUSE_RIGHT); break;
                        case 2: Mouse.release(MOUSE_MIDDLE); break;
                    }
                }
            }
            break;
        }
        
        case CMD_MOUSE_WHEEL: {
            // 数据: vertical(int8), horizontal(int8)
            if (len >= 2) {
                int8_t vWheel = (int8_t)data[0];
                int8_t hWheel = (int8_t)data[1];
                Mouse.move(0, 0, vWheel);
            }
            break;
        }
        
        case CMD_MOUSE_COMBINED: {
            // 数据: dx(int16), dy(int16), buttons(uint8), wheel_v(int8)
            if (len >= 6) {
                int16_t dx = (int16_t)(data[0] | (data[1] << 8));
                int16_t dy = (int16_t)(data[2] | (data[3] << 8));
                uint8_t buttons = data[4];
                int8_t wheelV = (int8_t)data[5];
                enqueueMouseMove((int8_t)dx, (int8_t)dy, buttons, wheelV);
            }
            break;
        }
        
        case CMD_HEARTBEAT: {
            // 心跳 - 回复确认
            sendResponse(RESP_HEARTBEAT_ACK, STATUS_OK, nullptr, 0);
            break;
        }
        
        case CMD_STATUS_QUERY: {
            // 状态查询 - 回复设备状态
            uint8_t statusData[4];
            statusData[0] = FIRMWARE_VERSION;
            statusData[1] = 0xE8;  // 1000 Hz low byte (0x03E8)
            statusData[2] = 0x03;  // 1000 Hz high byte
            statusData[3] = queueCount;
            sendResponse(RESP_STATUS, STATUS_OK, statusData, 4);
            break;
        }
        
        case CMD_CONFIG_SET: {
            // 配置设置
            if (len >= 3) {
                uint8_t key = data[0];
                int16_t value = (int16_t)(data[1] | (data[2] << 8));
                // 处理配置
                Serial.printf("Config: key=%d, value=%d\n", key, value);
                sendResponse(RESP_STATUS, STATUS_OK, nullptr, 0);
            }
            break;
        }
        
        default: {
            // 未知指令
            Serial.printf("Unknown command: 0x%02X\n", cmd);
            uint8_t errData[1] = {cmd};
            sendResponse(RESP_ERROR, STATUS_ERROR, errData, 1);
            break;
        }
    }
}

// ===== 发送响应 =====
void sendResponse(uint8_t respType, uint8_t status, uint8_t* data, uint16_t len) {
    uint8_t packet[320];
    uint16_t idx = 0;
    
    // 包头
    packet[idx++] = HEADER_TX_0;
    packet[idx++] = HEADER_TX_1;
    
    // 版本
    packet[idx++] = PROTO_VERSION;
    
    // 响应类型
    packet[idx++] = respType;
    
    // 状态
    packet[idx++] = status;
    
    // 数据长度
    packet[idx++] = (uint8_t)(len & 0xFF);
    packet[idx++] = (uint8_t)((len >> 8) & 0xFF);
    
    // 数据
    if (data != nullptr && len > 0) {
        memcpy(&packet[idx], data, len);
        idx += len;
    }
    
    // 校验和
    uint8_t checksum = 0;
    for (uint16_t i = 0; i < idx; i++) {
        checksum += packet[i];
    }
    packet[idx++] = checksum;
    
    // 发送
    Serial.write(packet, idx);
    Serial.flush();
    txCount++;
}

// ===== 鼠标指令队列 =====
void enqueueMouseMove(int8_t dx, int8_t dy, uint8_t buttons, int8_t wheelV) {
    // 如果队列满，覆盖最旧的
    if (queueCount >= QUEUE_SIZE) {
        queueTail = (queueTail + 1) % QUEUE_SIZE;
        queueCount--;
    }
    
    MouseState* state = &cmdQueue[queueHead];
    state->dx = dx;
    state->dy = dy;
    state->buttons = buttons;
    state->wheelV = wheelV;
    
    queueHead = (queueHead + 1) % QUEUE_SIZE;
    queueCount++;
}

bool dequeueMouseMove(MouseState* state) {
    if (queueCount == 0 || state == nullptr) {
        return false;
    }
    
    *state = cmdQueue[queueTail];
    queueTail = (queueTail + 1) % QUEUE_SIZE;
    queueCount--;
    return true;
}

// ===== 刷新鼠标队列 =====
void flushMouseQueue() {
    // 批量处理队列中的指令，合并连续移动
    int16_t totalDx = 0;
    int16_t totalDy = 0;
    uint8_t finalButtons = 0;
    int8_t finalWheel = 0;
    uint8_t processed = 0;
    
    MouseState state;
    while (dequeueMouseMove(&state) && processed < 8) {
        totalDx += state.dx;
        totalDy += state.dy;
        finalButtons |= state.buttons;
        finalWheel += state.wheelV;
        processed++;
    }
    
    if (processed > 0) {
        // 限制范围 (HID鼠标报告使用有符号8位)
        totalDx = constrain(totalDx, -127, 127);
        totalDy = constrain(totalDy, -127, 127);
        finalWheel = (int8_t)constrain(finalWheel, -127, 127);
        
        // 发送鼠标移动
        if (totalDx != 0 || totalDy != 0 || finalWheel != 0) {
            Mouse.move((int8_t)totalDx, (int8_t)totalDy, finalWheel);
        }
        
        // 处理按键
        if (finalButtons & 0x01) {
            Mouse.click(MOUSE_LEFT);
        }
        if (finalButtons & 0x02) {
            Mouse.click(MOUSE_RIGHT);
        }
    }
}
