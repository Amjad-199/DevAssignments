def mad_libs_generator():

    # Define a list of prompts for user input
    prompts = [
        "Enter an adjective: ",
        "Enter a noun: ",
        "Enter the name of a country: ",
        "Enter a type of building: ",
        "Enter an item that you like: ",
        "Enter another item that you like: ",
        "Enter an emotion: ",
        "Enter a verb ending with -ing: ",
        "Enter the name of a famous person: ",
        "Enter an animal: "
    ]

    # Append user input for each prompt to the list
    response = []
    # Prompt_index will be increased by 1 each time user responds
    prompt_index = 0

    # For loop to ask for user input for each prompt
    for prompt in prompts:
        while True:
            # prompt the user for a series of inputs a la Mad Libs
            user_input = input(prompts[prompt_index]).strip()
            
            if user_input:
                response.append(user_input)
                prompt_index += 1
                break
            else:
                print("Input cannot be blank. Please try again.")

    # Assign the responses to the meaningful variable names
    adjective, noun, country, building, prize_1, prize_2, emotion, verb, famous_person, animal = response

    # Pre-made story template with user input
    story = f"""

    Your Mad Libs story:

    Once upon a time there was a {adjective} {noun}. One day, the {noun} went to {country} and discovered a hidden {building} with one {prize_1} and one {prize_2}.
    Upon seeing the {prize_1} and the {prize_2}, the {noun} felt {emotion} and started {verb}.
    {famous_person} then walked in and saw the {noun} {verb}, and told the {noun} that if they decided to stay in the {building}, they could keep the {prize_1} and the {prize_2} but would need to fight against a {animal}.
    If the {noun} decided to leave then they would go back to {country} and never see a {prize_2} again.
    Since the {noun} didn't want to fight against a {animal}, the {noun} decided to give up the {prize_2} and return to {country} with a {prize_1}.

    """

    # print out the full story at the end with the input included
    print(story)

# Run the Mad Libs generator
mad_libs_generator()