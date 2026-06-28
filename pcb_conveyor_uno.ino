
const int sensorPin = 2; 
const int motorIN1  = 4;
const int motorIN2  = 5;

bool conveyorRunning  = false;
bool pcbDetected      = false;
bool waitingForResume = false;

// Sensor bypass: ignore IR sensor for BYPASS_DURATION ms after inspection
const unsigned long BYPASS_DURATION = 2000;
bool     sensorBypassed   = false;
unsigned long bypassStart = 0;

const unsigned long DEBOUNCE_MS = 50;
unsigned long sensorTriggeredAt = 0;
bool sensorArmed = false;

String inputBuffer = "";

void setup() {
  Serial.begin(115200);
  pinMode(sensorPin, INPUT_PULLUP);
  pinMode(motorIN1, OUTPUT);
  pinMode(motorIN2, OUTPUT);
  stopConveyor();
  Serial.println("READY");
}

void loop() {
  readSerial();
  checkBypassTimer();
  checkSensor();
}

// Serial 
void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      inputBuffer.trim();
      if (inputBuffer == "START") {
        waitingForResume = false;
        pcbDetected      = false;
        startConveyor();
      } else if (inputBuffer == "STOP") {
        waitingForResume = false;
        stopConveyor();
      } else if (inputBuffer == "SENSOR_BYPASS") {
        // Disable sensor for 2s so current PCB can clear past the sensor
        sensorBypassed    = true;
        bypassStart       = millis();
        sensorArmed       = false;
        pcbDetected       = false;
        waitingForResume  = false;
      }
      inputBuffer = "";
    } else {
      inputBuffer += c;
    }
  }
}

// Bypass timer 
void checkBypassTimer() {
  if (sensorBypassed && (millis() - bypassStart >= BYPASS_DURATION)) {
    sensorBypassed = false;   // re-arm sensor
  }
}

//  IR Sensor
void checkSensor() {

  if (!conveyorRunning || waitingForResume || sensorBypassed) return;

  bool objectPresent = (digitalRead(sensorPin) == LOW);

  if (objectPresent && !sensorArmed) {
    sensorTriggeredAt = millis();
    sensorArmed = true;
  } else if (objectPresent && sensorArmed) {
    if ((millis() - sensorTriggeredAt >= DEBOUNCE_MS) && !pcbDetected) {
      pcbDetected      = true;
      waitingForResume = true;
      stopConveyor();
      Serial.println("PCB_DETECTED");
    }
  } else {
    sensorArmed = false;
  }
}

// Motor 
void startConveyor() {
  digitalWrite(motorIN1, HIGH);
  digitalWrite(motorIN2, LOW);
  conveyorRunning = true;
  Serial.println("CONVEYOR_ON");
}

void stopConveyor() {
  digitalWrite(motorIN1, LOW);
  digitalWrite(motorIN2, LOW);
  conveyorRunning = false;
  Serial.println("CONVEYOR_OFF");
}
