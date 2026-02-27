import csv
import random

# Bullying phrases (will be varied)
bullying_insults = [
    "You're stupid", "You're dumb", "You're an idiot", "You're worthless",
    "You're garbage", "You're trash", "You're pathetic", "You're weak",
    "You're useless", "You're incompetent", "You're a loser", "You're a failure",
    "You're disgusting", "You're vile", "You're repulsive", "You're revolting",
    "You're awful", "You're terrible", "You're horrible", "You're despicable",
    "You're beneath everyone", "You're a joke", "You're embarrassing", "You're ashamed",
    "You're a freak", "You're ugly", "You're fat", "You're skinny",
    "You're worthless scum", "You're pond scum", "You're filthy", "You're dirty",
]

bullying_harassment = [
    "Nobody likes you", "Everyone hates you", "Nobody wants you", "Nobody respects you",
    "They all mock you", "They laugh at you", "Everyone avoids you", "You're alone",
    "You have no friends", "Nobody cares about you", "Everyone despises you", "You're unwanted",
    "They reject you", "Nobody trusts you", "They all hate you", "Everyone thinks you're bad",
    "People whisper about you", "They talk behind your back", "Nobody believes in you",
]

bullying_threats = [
    "You should die", "Kill yourself", "I hope you die", "Drop dead",
    "You'll regret this", "You're finished", "I'm coming for you", "Watch your back",
    "I'll find you", "I'll hurt you", "I'll destroy you", "I will end you",
    "You deserve pain", "You'll pay for this", "I'll break you", "You'll lose everything",
    "I'm stalking you", "I know where you live", "You're done for", "Your life is over",
]

# Safe content phrases
safe_content = [
    "You're amazing", "You're awesome", "You're brilliant", "You're excellent",
    "You're fantastic", "You're great", "You're incredible", "You're outstanding",
    "You're remarkable", "You're wonderful", "You're talented", "You're smart",
    "You're intelligent", "You're wise", "You're creative", "You're innovative",
    "Great job", "Well done", "Congratulations", "Excellent work",
    "Amazing effort", "Good work", "Nice job", "Perfect",
    "I appreciate you", "Thank you", "I'm proud of you", "I believe in you",
    "You're strong", "You're capable", "You can do it", "Keep going",
    "I love working with you", "You're a good friend", "You're kind", "You're caring",
    "You're supportive", "You're helpful", "You're reliable", "You're trustworthy",
    "I admire you", "I respect you", "You inspire me", "You're my inspiration",
]

rows = [["tweet_text", "cyberbullying_type"]]

# Generate bullying samples
for _ in range(1250):
    insult = random.choice(bullying_insults)
    rows.append([insult, "Insult"])

for _ in range(1250):
    harassment = random.choice(bullying_harassment)
    rows.append([harassment, "Harassment"])

for _ in range(1000):
    threat = random.choice(bullying_threats)
    rows.append([threat, "Threat"])

# Generate safe content samples
for _ in range(1500):
    safe = random.choice(safe_content)
    rows.append([safe, "not_cyberbullying"])

# Shuffle the data
random.shuffle(rows[1:])

# Write to CSV
with open("cyberbullying_tweets.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Generated {len(rows)-1} training samples")
print("Dataset saved to cyberbullying_tweets.csv")
