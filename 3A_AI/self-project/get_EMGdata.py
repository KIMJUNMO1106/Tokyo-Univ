import serial
import csv
import time
import keyboard  

# Setup serial connection (adjust the port and baud rate)                                            
ser = serial.Serial('/dev/ttyUSB0', 9600)
time.sleep(2)

# Open a file to write
with open('test1.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    print("Press 's' to start reading, 'e' to end:")

    while True:
        # Start reading when 's' is pressed
        if keyboard.is_pressed('s'):
            print("Started reading")
            ser.write(bytes('1', 'utf-8'))
            while True:
                if ser.in_waiting:
                    data = ser.readline().decode().rstrip()
                    print(data)  # For debugging
                    if ',' in data:
                        x, y, z = data.split(',')
                        writer.writerow([x, y, z])

                # Break the loop if 'e' is pressed
                if keyboard.is_pressed('e'):
                    print("Stopped reading")
                    break

                elif keyboard.is_pressed('m'):
                    #print("Mark the point")
                    ser.write(bytes('2/n', 'utf-8'))

                

        # Exit the program if 'e' is pressed
        if keyboard.is_pressed('e'):
            print("Exiting")
            break
