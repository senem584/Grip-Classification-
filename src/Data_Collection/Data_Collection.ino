#define SAMPLE_RATE 500
#define BAUD_RATE 115200
#define INPUT_PIN 33
#define BUFFER_SIZE 128
#define MAX_SAMPLES 5000  // Stop after 5000 samples

int circular_buffer[BUFFER_SIZE];
int data_index = 0;
int sum = 0;
unsigned long sample_count = 0;  // Track number of samples collected
bool collecting = true;          // Flag to stop data collection

void setup() {
  Serial.begin(BAUD_RATE);
  analogReadResolution(12);  // ESP32 supports up to 12 bits (4096 levels)
}

void loop() {
  if (!collecting) return;

  static unsigned long last_sample_time = 0;
  unsigned long now = millis();

  if (now - last_sample_time >= 1000 / SAMPLE_RATE) {
    last_sample_time = now;

    int sensor_value = analogRead(INPUT_PIN);
    int signal = EMGFilter(sensor_value);
    int envelop = getEnvelop(abs(signal));
    Serial.print(signal);
    Serial.print(",");
    Serial.println(envelop);

    sample_count++;
    if (sample_count >= MAX_SAMPLES) {
      collecting = false;
      Serial.println("Data collection complete.");
    }
  }
}


int getEnvelop(int abs_emg) {
  sum -= circular_buffer[data_index];
  sum += abs_emg;
  circular_buffer[data_index] = abs_emg;
  data_index = (data_index + 1) % BUFFER_SIZE;
  return (sum / BUFFER_SIZE) * 2;
}

// Band-Pass Butterworth IIR filter
float EMGFilter(float input) {
  float output = input;

  {
    static float z1, z2;
    float x = output - 0.05159732 * z1 - 0.36347401 * z2;
    output = 0.01856301 * x + 0.03712602 * z1 + 0.01856301 * z2;
    z2 = z1;
    z1 = x;
  }

  {
    static float z1, z2;
    float x = output - -0.53945795 * z1 - 0.39764934 * z2;
    output = 1.00000000 * x + -2.00000000 * z1 + 1.00000000 * z2;
    z2 = z1;
    z1 = x;
  }

  {
    static float z1, z2;
    float x = output - 0.47319594 * z1 - 0.70744137 * z2;
    output = 1.00000000 * x + 2.00000000 * z1 + 1.00000000 * z2;
    z2 = z1;
    z1 = x;
  }

  {
    static float z1, z2;
    float x = output - -1.00211112 * z1 - 0.74520226 * z2;
    output = 1.00000000 * x + -2.00000000 * z1 + 1.00000000 * z2;
    z2 = z1;
    z1 = x;
  }

  return output;
}
