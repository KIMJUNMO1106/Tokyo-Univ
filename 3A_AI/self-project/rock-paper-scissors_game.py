import time
import random
import serial
import keyboard
from collections import deque
import numpy as np
import openpyxl
from sklearn.model_selection import train_test_split
import pygame

filename = "jyanken.xlsx"
book = openpyxl.load_workbook(filename)
sheet=book.worksheets[0]


data = []
for row in sheet.rows:
  data.append([
      row[0].value,
      row[1].value,
      row[2].value,
      row[3].value,
      row[4].value,
      row[5].value,
      row[6].value,
      row[7].value,
      row[8].value,
      row[9].value,
      row[10].value


  ])
data = np.array(data)
data_subset = data[1:,:]
# Splitting labels (y) and features (X)
x = data_subset[:, 1:].astype(float)
x = np.round(x,2)  # Convert feature data to float
y = data_subset[:, 0].astype(int)  # Labels

# Splitting into train and test sets (90% train, 10% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01)

# Check the shapes
x_train.shape, x_test.shape, y_train.shape, y_test.shape
print(y_test)


def softmax(z):
    e_z = np.exp(z)
    return e_z / e_z.sum(axis=1, keepdims=True)


class SoftmaxRegression:

    def __init__(self, learning_rate=0.01, epochs=1000, num_classes=6):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.num_classes = num_classes

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, self.num_classes))
        #print(self.num_classes)
        y_cat = np.eye(self.num_classes)[y]

        #print(np.shape(y))
        #print(y_cat)

        for _ in range(self.epochs):
            scores = np.dot(X, self.weights)
            y_pred = softmax(scores)
            self.weights -= self.learning_rate * (1/n_samples) * np.dot(X.T, (y_pred - y_cat))


    def predict(self, X):
        scores = np.dot(X, self.weights)
        y_pred = softmax(scores)
        return np.argmax(y_pred, axis=1)

model = SoftmaxRegression(0.01, 5000)
model.fit(x_train, y_train)


def rms(signal):
    return np.sqrt(np.mean(signal**2))

def variance(signal):
    return np.var(signal)

def mav(signal):
    return np.mean(np.abs(signal))

def ssc(signal):
    slope_changes = np.diff(np.sign(np.diff(signal)))
    return np.sum(slope_changes != 0)

def sigma_diff(signal):
    diffs = np.diff(signal)
    return np.std(diffs)

def calculate_mean(signal):
    mean = np.mean(signal)

def calculate_features(batch):
    features = []
    for signal in zip(*batch):
        signal_array = np.array(signal)
        features.extend([
            rms(signal_array),
            variance(signal_array),
            mav(signal_array),
            ssc(signal_array),
            sigma_diff(signal_array)
        ])
    return features

def calculate_mean(batch):
    features = []
    for signal in zip(*batch):
        signal_array = np.array(signal)
        features.extend([
            mav(signal_array)
        ])
        
    return features


def calculate_difference(data_queue, new_data):
    if len(data_queue) == 14: #14
        old_data = data_queue.popleft()  # Remove the oldest data
        return [new_data[i] - old_data[i] for i in range(len(new_data))]
    return None  # Not enough data yet


pygame.init()
# Set up the game window
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Load gesture images
gestures = {
    '1': pygame.image.load('rock.png'),
    '2': pygame.image.load('paper.png'),
    '3': pygame.image.load('scissors.png')
}

def determine_outcome(player, computer):
    if player == computer:
        return 'draw'
    elif (player == '1' and computer == '3') or \
         (player == '2' and computer == '1') or \
         (player == '3' and computer == '2'):
        return 'win'
    else:
        return 'lose'
    
# Setup serial connection
ser = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)


print("Press 's' to start reading, 'e' to end:")

data_queue = deque(maxlen=14)  # One extra to calculate the difference
difference_queue = deque(maxlen=13) 
count=0
judge=0
countt=0
result=[]
current_gesture=None
flag = False
win, draw, lose=0,0,0
font = pygame.font.Font(None, 36)
font2 = pygame.font.Font(None, 90)
running = True
while running:
    # Check for Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                print("Started reading")
                ser.write(bytes('1', 'utf-8'))
                
            elif event.key == pygame.K_e:
                print("Exiting")
                running = False
                break

    if ser.in_waiting:
        data = ser.readline().decode().rstrip()
        if ',' in data:
            x, y = map(float, data.split(','))
            new_data = [x, y]
            data_queue.append(new_data)  # Store the new data
            difference = calculate_difference(data_queue, new_data)
            print(difference)
            if difference:
                difference_queue.append(difference)
                if len(difference_queue) == 13:
                    mean = calculate_mean(difference_queue)
                    if judge == 0:
                        screen.fill((0, 0, 0))  
                        wait_text = font.render("Please wait...", True, (255, 255, 255))  # White text
                        text_rect = wait_text.get_rect(center=(screen_width // 2, screen_height // 2))
                        screen.blit(wait_text, text_rect)
                        pygame.display.update()
                    
                        if abs(difference[0])<3 and abs(difference[1])<3 and mean[0]<2.5 and mean[1]<2.5:
                            count+=1
                        elif abs(difference[0])>3 or abs(difference[1])>3 or mean[0]>2.5 or mean[1]>2.5:
                            count=0

                    print(mean)
                    print("count",count)
                    print("judge", judge)
                    print("countt", countt)
                    print("result", result)

                    if count>50:
                      wait_text = font.render("Please wait...", True, (0, 0, 0))  # White text
                      text_rect = wait_text.get_rect(center=(screen_width // 2, screen_height // 2))
                      screen.blit(wait_text, text_rect)
                      pygame.display.update()
                      current_gesture = random.choice(list(gestures.keys()))
                      gesture_image = gestures[current_gesture]
                      screen.blit(gesture_image, (170, 200)) 
                      pygame.display.update()
                      judge=2 
                      count = 0
                      time.sleep(1)
                                            
                    

                    if judge ==2 and  (abs(difference[0])>4 or abs(difference[1])>5):
                        countt+=1

                    elif judge ==2 and countt>2:
                        features = calculate_features(difference_queue)
                        features = np.array(features).astype(float)
                        features_data = np.round(features,2)
                        features_data = np.reshape(features_data,(1,-1))
                        print("Features:", features_data)
                        predicted_class = model.predict(features_data)
                        print("Predicted Class:", predicted_class)
                        result.extend(predicted_class)
                        answer_text = font.render(f"Your answer is", True, (255, 255, 255))  
                        answer_rect = answer_text.get_rect(topright=(screen_width - 140, 120))  
                        screen.blit(answer_text, answer_rect)
                        predicted_gesture = str(predicted_class[0])
                        predicted_image = gestures[predicted_gesture]
                        screen.blit(predicted_image, (470, 200))  # Adjust the position as needed
                        pygame.display.update()
                        time.sleep(1)
                        countt = 0
                        judge = 0
                        outcome = determine_outcome(predicted_gesture, current_gesture)
                        if outcome == 'win':
                            win += 1
                        elif outcome == 'draw':
                            draw += 1
                        else:
                            lose += 1

                            # Display the outcome message
                        outcome_text = font2.render(outcome.capitalize(), True, (255, 255, 255))
                        outcome_rect = outcome_text.get_rect(center=(screen_width // 2, 500))
                        screen.blit(outcome_text, outcome_rect)
                        pygame.display.update()
                        time.sleep(1)
                        judge = 0
                        countt = 0
                            
    win_text = font.render(f"Win: {win}", True, (255, 255, 255))  
    win_rect = win_text.get_rect(topright=(screen_width - 20, 20))  
    screen.blit(win_text, win_rect)
    draw_text = font.render(f"Draw: {draw}", True, (255, 255, 255))  
    draw_rect = draw_text.get_rect(topright=(screen_width - 20, 40))
    screen.blit(draw_text, draw_rect)
    lose_text = font.render(f"Lose: {lose}", True, (255, 255, 255))
    lose_rect = lose_text.get_rect(topright=(screen_width - 20, 60))  
    screen.blit(lose_text, lose_rect)

    pygame.display.flip()
