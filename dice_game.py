import random 

print("Welcome to the dice-rolling simulator. Begin by rolling the die, you can then continue to roll as many times as you like. ")

dice = [1,2,3,4,5,6]

while True:
    first_roll = input("Please roll the die by typing in roll (Roll):").lower()
    roll = random.choice(dice)
    if first_roll == "roll":
        print(f"You have rolled a {roll}")
        break
    else:
        print("That is not a valid response")

while True:
    ask = input("Would you like to roll again?(yes/no):").lower()
    roll = random.choice(dice)
    if ask == "no":
        print("Thanks for playing!")
        break
    elif ask == "yes":
        print(f"You have rolled a {roll}")
        continue
    else:
        print("That is not a valid response")