templates = {
    "few_shot_cot_composition": """Answer the questions below step by step. Always conclude with the sentence 'So the final answer is:' followed by the final numerical answer.


Question: You have a buffet plate that is 1.3 cm thick, a saucer that is 0.7 cm thick, a dinner plate that is 1.65 cm thick, and a dessert plate that is 0.93 cm thick.
You want to stack them by decreasing diameter, the widest at the bottom. What will be the combined height of the two plates at the top of the stack, in centimeters?

Answer: For the plates to be stacked by decreasing diameter, the stacking order (from bottom to top) is:
Buffet plate (1.3 cm thick), dinner plate (1.65 cm thick), dessert plate (0.93 cm thick), saucer (0.7 cm thick).
Hence the two plates at the top of the stack are the dessert plate and the saucer, and their combined height is 0.93 + 0.7 = 1.63 cm.
So the final answer is: 1.63


Question: Kuala Lumpur has 4500 restaurants, Albuquerque has 653 restaurants, Dubai has 13,000 restaurants. How many restaurants do the Asian cities have, combined?

Answer: Kuala Lumpur and Dubai are in Asia, whereas Albuquerque is in America.
So the Asian cities are Kuala Lumpur and Dubai.
Kuala Lumpur has 4500 restaurants and Dubai has 13,000. So combined they have 4500+13000=17500 restaurants.
So the final answer is: 17500


Question: {question}

Answer: """,
    "few_shot_cot_commonsense": """Answer the questions below step by step. Always conclude with the sentence 'So the final answer is:' followed by the final answer.


Question: You have a buffet plate, a saucer, a dinner plate, and a dessert plate.
You want to stack them by decreasing diameter, the widest at the bottom. Which two plates are at the top of the stack?

Answer: For the plates to be stacked by decreasing diameter, the stacking order (from bottom to top) is:
Buffet plate, dinner plate, dessert plate, saucer.
Hence the two plates at the top of the stack are the dessert plate and the saucer.
So the final answer is: dessert plate and saucer


Question:  Kuala Lumpur has 4500 restaurants, Albuquerque has 653 restaurants, Dubai has 13,000 restaurants. Which of these are Asian cities?

Answer: Kuala Lumpur and Dubai are in Asia, whereas Albuquerque is in America. So the Asian cities are Kuala Lumpur and Dubai.
So the final answer is: Kuala Lumpur and Dubai


Question: {question}

Answer: """,
    "few_shot_cot_math": """Answer the questions below step by step. Always conclude with the sentence 'So the final answer is:' followed by the final numerical answer.


Question: You have a buffet plate that is 1.3 cm thick, a saucer that is 0.7 cm thick, a dinner plate that is 1.65 cm thick, and a dessert plate that is 0.93 cm thick.
You want to stack them by decreasing diameter, the widest at the bottom. What will be the combined height of the dessert plate and the saucer, in centimeters?

Answer: The dessert plate is 0.93 cm thick, and the saucer is 0.7 cm thick.
Hence their combined height is 0.93 + 0.7 = 1.63 cm.
So the final answer is: 1.63


Kuala Lumpur has 4500 restaurants, Albuquerque has 653 restaurants, Dubai has 13,000 restaurants. How many restaurants do Kuala Lumpur and Dubai have, combined?

Answer: Kuala Lumpur has 4500 restaurants and Dubai has 13,000. So combined they have 4500+13000=17500 restaurants.
So the final answer is: 17500


Question: {question}

Answer: """,
}
