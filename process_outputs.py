import os


for scnario in os.listdir("outputs"):
    with open(f"outputs/{scnario}/{scnario}.rou.xml", 'w') as f:
        pass