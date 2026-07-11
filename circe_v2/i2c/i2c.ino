#include <Wire.h>

const uint8_t ADDR = 0x68;
// Register map
const uint8_t PWR_MGMT_1 = 0x6B, ACCEL_CONFIG = 0x1C, GYRO_CONFIG = 0x1B, ACCEL_XOUT_H = 0x3B;

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADDR);
  Wire.write(reg); Wire.write(val);
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  while(!Serial) {}
  Wire.begin(15, 14);
  Wire.setClock(400000);

  writeReg(PWR_MGMT_1, 0x80);  delay(100);   // reset
  writeReg(PWR_MGMT_1, 0x01);  delay(100);   // wake, best clock (PLL)
  writeReg(ACCEL_CONFIG, 0x00);               // +/- 2 g
  writeReg(GYRO_CONFIG, 0x00);                // +/- 250 deg/s
  Serial.println("ax(m/s^2)\tay\taz\tgx(rad/s)\tgy\tgz\ttemp(C)");
}

void loop() {
  // Burst-read 14 bytes: accel(6), temp(2), gyro(6)
  Wire.beginTransmission(ADDR);
  Wire.write(ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom((int)ADDR, 14);
  if (Wire.available() < 14) return;

  int16_t ax = (Wire.read()<<8)|Wire.read();
  int16_t ay = (Wire.read()<<8)|Wire.read();
  int16_t az = (Wire.read()<<8)|Wire.read();
  int16_t t  = (Wire.read()<<8)|Wire.read();
  int16_t gx = (Wire.read()<<8)|Wire.read();
  int16_t gy = (Wire.read()<<8)|Wire.read();
  int16_t gz = (Wire.read()<<8)|Wire.read();

  const float A = 9.80665f/16384.0f;          // +/-2g  -> m/s^2
  const float G = (1.0f/131.0f)*(PI/180.0f);  // +/-250 dps -> rad/s
  float tempC = t/333.87f + 21.0f;            // MPU6500 datasheet formula

  Serial.print(ax*A,3); Serial.print("\t"); Serial.print(ay*A,3); Serial.print("\t"); Serial.print(az*A,3); Serial.print("\t");
  Serial.print(gx*G,3); Serial.print("\t"); Serial.print(gy*G,3); Serial.print("\t"); Serial.print(gz*G,3); Serial.print("\t");
  Serial.println(tempC,2);
  delay(50);
}