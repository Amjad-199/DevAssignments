import random 

print("Welcome to Guess the Number! ")

the_number = random.randint(1, 100)

while True:
    try:
            guess = int(input("Guess a number between 1 and 100"))
    except ValueError:
            print("Invalid input. Please enter a valid integer.")
            continue
    if guess > the_number:
        print(f"Wrong! {guess} is too high!")
    elif guess < the_number:
        print(f"Wrong! {guess} is too low!")
    elif guess == the_number:
        print(f"Well done! You have correctly guessed the number {the_number}")
        break