#include <Arduino.h>
#include <TFT_eSPI.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "grip_model_s.h"
#include <math.h>

#define EMG_PIN 33
#define SAMPLING_FREQ 500
#define WINDOW_SIZE 64  // Changed to power of 2 for FFT
#define NUM_FEATURES 24
#define DISPLAY_BL_PIN 4
#define ENVELOPE_BUFFER_SIZE 128
#define DISPLAY_UPDATE_MS 5000
#define TFT_BLACK 0x0000
#define TFT_WHITE 0xFFFF
#define TFT_CYAN  0x07FF
#define TFT_DARKGREY 0x7BEF
#define EMG_THRESHOLD 50
#define CONFIDENCE_THRESHOLD 0.6f
#define OVERLAP 25
#define SLIDING_BUFFER_SIZE (WINDOW_SIZE * 2)

// Quantization parameters
const float INPUT_SCALE = 0.033870019018650055f;
const int8_t INPUT_ZERO_POINT = -34;
const float OUTPUT_SCALE = 0.00390625f;
const int8_t OUTPUT_ZERO_POINT = -128;

// Scaler parameters
const float scaler_mean[NUM_FEATURES] = {
    75.893293, 93.695606, 14231.133197, 3794.664651, 15.235887, 4049.460013, 
    19.481519, 48.675403, 47.793002, 9381.111688, 99.732806, 95.346102, 
    150.446747, 150.567069, 62.273538, 7522.337366, 0.000000, 48.588710, 
    4.631720, 19.104503, 150.324950, 7999.607066, 4.225365, 0.000000
};

const float scaler_scale[NUM_FEATURES] = {
    59.042009, 73.892020, 22690.784478, 2952.100455, 3.390095, 3586.373565,
    3.020037, 0.610295, 41.171668, 8717.177357, 6.709998, 11.072275,
    114.137582, 114.251664, 189.316755, 5706.879113, 1.000000, 39.077779,
    4.277597, 10.216482, 114.023770, 6151.288014, 2.487493, 1.000000
};

// Grip class labels
const char* grip_names[] = {
    "Tip", "Spherical", "Palmer", "Cylindrical", "Hook", "Lateral"
};
const int num_grip_classes = sizeof(grip_names) / sizeof(grip_names[0]);

// 3. Class definitions
class CircularBuffer {
private:
    float data[SLIDING_BUFFER_SIZE];
    int head = 0;
    bool is_full = false;

public:
    void push(float value) {
        data[head] = value;
        head = (head + 1) % SLIDING_BUFFER_SIZE;
        if (head == 0) is_full = true;
    }

    bool isFull() const { return is_full; }
    
    const float* getData() const { return data; }
    
    void getWindow(float* window) {
        if (!window) return;
        int start = (head - WINDOW_SIZE + SLIDING_BUFFER_SIZE) % SLIDING_BUFFER_SIZE;
        for (int i = 0; i < WINDOW_SIZE; i++) {
            window[i] = data[(start + i) % SLIDING_BUFFER_SIZE];
        }
    }

    bool hasNewWindow() const {
        static int last_head = 0;
        int samples_since_last = (head - last_head + SLIDING_BUFFER_SIZE) % SLIDING_BUFFER_SIZE;
        if (samples_since_last >= (WINDOW_SIZE - OVERLAP)) {
            last_head = head;
            return true;
        }
        return false;
    }
};
// Global variables
TFT_eSPI tft = TFT_eSPI();
CircularBuffer emg_raw_buffer;
CircularBuffer emg_env_buffer;
int envelope_buffer[ENVELOPE_BUFFER_SIZE] = {0};
int envelope_index = 0;
int envelope_sum = 0;
bool model_loaded = false;

// TFLite globals
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

// Performance monitoring
struct PerformanceMetrics {
    unsigned long last_inference_time = 0;
    int inference_count = 0;
    float avg_inference_time = 0;
} metrics;

// Feature extraction
float calculateMAV(const float* buffer, int size);
float calculateRMS(const float* buffer, int size);
float calculateVAR(const float* buffer, int size);
float calculateIEMG(const float* buffer, int size);
int calculateZC(const float* buffer, int size, float threshold = 0.01f);
float calculateWL(const float* buffer, int size);
int calculateSSC(const float* buffer, int size, float threshold = 0.01f);
int calculateWAMP(const float* buffer, int size, float threshold = 0.02f);
float calculateLOG_D(const float* buffer, int size);

// FFT and frequency domain
unsigned int bitReverse(unsigned int x, int bits);
void fft(float* real, float* imag, int N);
void calculateFrequencyFeatures(const float* signal, int N, float fs, float& totalPower, float& mnf, float& mdf);

// Signal processing
float EMGFilter(float input);
int calculateEnvelope(int abs_emg);
void extractFeatures(const float* raw_buffer, const float* env_buffer, float* features);

// Display and model
void updateDisplay(const char* grip_name, float confidence, const float* signal);
bool initializeModel();
bool isSignalValid(const float* buffer, int size);

// 6. Function implementations
// Feature extraction implementations
float calculateMAV(const float* buffer, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += abs(buffer[i]);
    }
    return sum / size;
}

float calculateRMS(const float* buffer, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += buffer[i] * buffer[i];
    }
    return sqrt(sum / size);
}

float calculateVAR(const float* buffer, int size) {
    float mean = 0;
    for (int i = 0; i < size; i++) {
        mean += buffer[i];
    }
    mean /= size;

    float variance = 0;
    for (int i = 0; i < size; i++) {
        float diff = buffer[i] - mean;
        variance += diff * diff;
    }
    return variance / (size - 1);
}

float calculateIEMG(const float* buffer, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += abs(buffer[i]);
    }
    return sum;
}

int calculateZC(const float* buffer, int size, float threshold) {
    int count = 0;
    for (int i = 0; i < size - 1; i++) {
        if ((buffer[i] > threshold && buffer[i + 1] < -threshold) ||
            (buffer[i] < -threshold && buffer[i + 1] > threshold)) {
            count++;
        }
    }
    return count;
}

float calculateWL(const float* buffer, int size) {
    float sum = 0;
    for (int i = 1; i < size; i++) {
        sum += abs(buffer[i] - buffer[i-1]);
    }
    return sum;
}

int calculateSSC(const float* buffer, int size, float threshold) {
    int count = 0;
    for (int i = 1; i < size - 1; i++) {
        float diff1 = buffer[i] - buffer[i-1];
        float diff2 = buffer[i+1] - buffer[i];
        if ((diff1 > 0 && diff2 < 0) || (diff1 < 0 && diff2 > 0)) {
            if (abs(diff1) >= threshold && abs(diff2) >= threshold) {
                count++;
            }
        }
    }
    return count;
}int calculateWAMP(const float* buffer, int size, float threshold) {
    int count = 0;
    for (int i = 0; i < size - 1; i++) {
        if (abs(buffer[i+1] - buffer[i]) > threshold) {
            count++;
        }
    }
    return count;
}

float calculateLOG_D(const float* buffer, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += log(abs(buffer[i]) + 1e-6);
    }
    return exp(sum / size);
}

// FFT implementations
unsigned int bitReverse(unsigned int x, int bits) {
    unsigned int result = 0;
    for (int i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

void fft(float* real, float* imag, int N) {
    int bits = log2(N);
    for (int i = 0; i < N; i++) {
        int j = bitReverse(i, bits);
        if (i < j) {
            float temp_real = real[i];
            float temp_imag = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = temp_real;
            imag[j] = temp_imag;
        }
    }

    for (int step = 2; step <= N; step <<= 1) {
        float angle = -2 * PI / step;
        float wr = cos(angle);
        float wi = sin(angle);

        for (int i = 0; i < N; i += step) {
            float w_real = 1;
            float w_imag = 0;

            for (int j = 0; j < step/2; j++) {
                int a = i + j;
                int b = i + j + step/2;

                float temp_real = w_real * real[b] - w_imag * imag[b];
                float temp_imag = w_real * imag[b] + w_imag * real[b];

                real[b] = real[a] - temp_real;
                imag[b] = imag[a] - temp_imag;
                real[a] = real[a] + temp_real;
                imag[a] = imag[a] + temp_imag;

                float temp = w_real * wr - w_imag * wi;
                w_imag = w_real * wi + w_imag * wr;
                w_real = temp;
            }
        }
    }
}

void calculateFrequencyFeatures(const float* signal, int N, float fs, float& totalPower, float& mnf, float& mdf) {
    float* real = new float[N];
    float* imag = new float[N];
    float* power = new float[N/2];
    
    if (!real || !imag || !power) {
        Serial.println("Memory allocation failed");
        if (real) delete[] real;
        if (imag) delete[] imag;
        if (power) delete[] power;
        return;
    }
    
    for (int i = 0; i < N; i++) {
        real[i] = signal[i];
        imag[i] = 0;
    }
    
    fft(real, imag, N);
    
    totalPower = 0;
    for (int i = 0; i < N/2; i++) {
        power[i] = sqrt(real[i]*real[i] + imag[i]*imag[i]);
        totalPower += power[i];
    }
    
    float freqSum = 0;
    for (int i = 0; i < N/2; i++) {
        float freq = i * fs / N;
        freqSum += freq * power[i];
    }
    mnf = freqSum / totalPower;
    
    float cumPower = 0;
    int mdfIndex = 0;
    for (int i = 0; i < N/2; i++) {
        cumPower += power[i];
        if (cumPower >= totalPower/2) {
            mdfIndex = i;
            break;
        }
    }
    mdf = mdfIndex * fs / N;
    
    delete[] real;
    delete[] imag;
    delete[] power;
}

// Signal processing implementations
float EMGFilter(float input) {
    static float z1a, z2a, z1b, z2b, z1c, z2c, z1d, z2d;
    float output = input;

    float x = output - 0.05159732f * z1a - 0.36347401f * z2a;
    output = 0.01856301f * x + 0.03712602f * z1a + 0.01856301f * z2a;
    z2a = z1a; z1a = x;

    x = output - -0.53945795f * z1b - 0.39764934f * z2b;
    output = x + -2.0f * z1b + z2b;
    z2b = z1b; z1b = x;

    x = output - 0.47319594f * z1c - 0.70744137f * z2c;
    output = x + 2.0f * z1c + z2c;
    z2c = z1c; z1c = x;

    x = output - -1.00211112f * z1d - 0.74520226f * z2d;
    output = x + -2.0f * z1d + z2d;
    z2d = z1d; z1d = x;

    return output;
}
int calculateEnvelope(int abs_emg) {
    envelope_sum -= envelope_buffer[envelope_index];
    envelope_sum += abs_emg;
    envelope_buffer[envelope_index] = abs_emg;
    envelope_index = (envelope_index + 1) % ENVELOPE_BUFFER_SIZE;
    return (envelope_sum / ENVELOPE_BUFFER_SIZE) * 2;
}

void extractFeatures(const float* raw_buffer, const float* env_buffer, float* features) {
    int idx = 0;
    
    // EMG1 features (raw signal)
    features[idx++] = calculateMAV(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateRMS(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateVAR(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateIEMG(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateZC(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateWL(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateSSC(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateWAMP(raw_buffer, WINDOW_SIZE);
    features[idx++] = calculateLOG_D(raw_buffer, WINDOW_SIZE);

    float totalPower1, mnf1, mdf1;
    calculateFrequencyFeatures(raw_buffer, WINDOW_SIZE, SAMPLING_FREQ, totalPower1, mnf1, mdf1);
    features[idx++] = totalPower1;
    features[idx++] = mnf1;
    features[idx++] = mdf1;

    // EMG2 features (envelope signal)
    features[idx++] = calculateMAV(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateRMS(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateVAR(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateIEMG(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateZC(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateWL(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateSSC(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateWAMP(env_buffer, WINDOW_SIZE);
    features[idx++] = calculateLOG_D(env_buffer, WINDOW_SIZE);

    float totalPower2, mnf2, mdf2;
    calculateFrequencyFeatures(env_buffer, WINDOW_SIZE, SAMPLING_FREQ, totalPower2, mnf2, mdf2);
    features[idx++] = totalPower2;
    features[idx++] = mnf2;
    features[idx++] = mdf2;
}

void updateDisplay(const char* grip_name, float confidence, const float* signal) {
    static char last_grip[20] = "";
    static float last_conf = -1;

    if (strcmp(last_grip, grip_name) != 0 || abs(last_conf - confidence) > 0.05f) {
        tft.fillRect(0, 0, tft.width(), 40, TFT_BLACK);
        tft.setCursor(0, 0);
        tft.println(grip_name);
        tft.print(int(confidence * 100));
        tft.println("%");
        
        strcpy(last_grip, grip_name);
        last_conf = confidence;
    }

    tft.fillRect(0, 120, tft.width(), 60, TFT_BLACK);
    tft.drawLine(0, 150, tft.width(), 150, TFT_DARKGREY);
    
    if (signal) {
        for (int i = 0; i < WINDOW_SIZE - 1; i++) {
            int x1 = map(i, 0, WINDOW_SIZE - 1, 0, tft.width());
            int x2 = map(i + 1, 0, WINDOW_SIZE - 1, 0, tft.width());
            int y1 = map(constrain(signal[i], -2000, 2000), -2000, 2000, 180, 120);
            int y2 = map(constrain(signal[i + 1], -2000, 2000), -2000, 2000, 180, 120);
            tft.drawLine(x1, y1, x2, y2, TFT_CYAN);
        }
    }
}

bool initializeModel() {
    const tflite::Model* model = tflite::GetModel(grip_model_s_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return false;
    }

    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return false;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    if (input->dims->data[1] != NUM_FEATURES) {
        Serial.println("Input dimension mismatch!");
        return false;
    }

    return true;
}

bool isSignalValid(const float* buffer, int size) {
    float sum = 0;
    for(int i = 0; i < size; i++) {
        sum += abs(buffer[i]);
    }
    float avg = sum / size;
    return avg > EMG_THRESHOLD;
}

void setup() {
    Serial.begin(115200);
    
    pinMode(DISPLAY_BL_PIN, OUTPUT);
    digitalWrite(DISPLAY_BL_PIN, HIGH);
    tft.init();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextSize(2);
    
    if (!initializeModel()) {
        tft.println("Model init failed!");
        while (1) delay(1000);
    }
    
    model_loaded = true;
    tft.println("Ready for EMG");
}

void loop() {
    static unsigned long last_sample_time = 0;
    static float window_buffer[WINDOW_SIZE];
    static float env_window_buffer[WINDOW_SIZE];
    unsigned long now = millis();

    if (now - last_sample_time >= 1000 / SAMPLING_FREQ) {
        last_sample_time = now;
        
        int raw_value = analogRead(EMG_PIN);
        float filtered = EMGFilter(raw_value);
        int env = calculateEnvelope(abs(filtered));
        
        emg_raw_buffer.push(filtered);
        emg_env_buffer.push(env);

        if (emg_raw_buffer.isFull() && emg_raw_buffer.hasNewWindow()) {
            emg_raw_buffer.getWindow(window_buffer);
            emg_env_buffer.getWindow(env_window_buffer);

            if (isSignalValid(window_buffer, WINDOW_SIZE)) {
                float features[NUM_FEATURES];
                extractFeatures(window_buffer, env_window_buffer, features);
                
                for(int i = 0; i < NUM_FEATURES; i++) {
                    float standardized = (features[i] - scaler_mean[i]) / scaler_scale[i];
                    input->data.int8[i] = (int8_t)(standardized / INPUT_SCALE + INPUT_ZERO_POINT);
                }

                if (interpreter->Invoke() == kTfLiteOk) {
                    float max_score = -1;
                    int prediction = -1;
                    
                    for (int i = 0; i < num_grip_classes; i++) {
                        float score = (output->data.int8[i] - OUTPUT_ZERO_POINT) * OUTPUT_SCALE;
                        if (score > max_score) {
                            max_score = score;
                            prediction = i;
                        }
                    }

                    if (max_score >= CONFIDENCE_THRESHOLD) {
                        updateDisplay(grip_names[prediction], max_score, window_buffer);
                    } else {
                        updateDisplay("Uncertain", max_score, window_buffer);
                    }
                }
            } else {
                updateDisplay("No Signal", 0, window_buffer);
            }
        }
    }
}




