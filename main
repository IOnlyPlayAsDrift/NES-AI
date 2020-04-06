import retro
import numpy as np
import cv2
import neat
import pickle

env = retro.make(game = "IceClimber-Nes", state = "Level1")


imgarray = []

resume = False
restorefile = "neat-checkpoint-5"

def eval_genomes(genomes, config):


    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()

        inx, iny, inc = env.observation_space.shape

        inx = int(inx/8)
        iny = int(iny/8)

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        score = 0
        score_max = 0
        birds = 0
        birds_max = 0
        bricks = 0
        bricks_max = 0
        eggplant = 0
        eggplant_max = 0
        ice = 0
        ice_max = 0
        lives = 3
        lives_max = 3

        done = False
        cv2.namedWindow("main", cv2.WINDOW_NORMAL)

        while not done:

            env.render()
            frame += 1
            scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            scaledimg = cv2.resize(scaledimg, (iny, inx))
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            cv2.imshow("main", scaledimg)
            cv2.waitKey(1)
            for x in ob:
                for y in x:
                    imgarray.append(y)

            nnOutput = net.activate(imgarray)

            ob, rew, done, info = env.step(nnOutput)
            imgarray.clear()
            
            score = info["score"]
            birds = info["birds_hit"]
            bricks = info["bricks_hit"]
            eggplant = info["eggplant_hit"]
            ice = info["ice_hit"]
            lives = info["lives"]
            
            if score > score_max:
                score_current = score - score_max
                print(score_current)
                fitness_current += score_current
                score_max = score

            if birds > birds_max:
                print("Added 100 Fitness for hitting a bird.")
                fitness_current += 100
                birds_max = birds

            if bricks > bricks_max:
                print("Added 10 Fitness for hitting a brick.")
                fitness_current += 10
                bricks_max = bricks

            if eggplant > eggplant_max:
                print("Added 50 Fitness for hitting an eggplant.")
                fitness_current += 50
                eggplant_max = eggplant

            if ice > ice_max:
                print("Added 10 Fitness for hitting ice.")
                fitness_current += 10
                ice_max = ice

            if lives < lives_max:
                print("Subtracted 25 Fitness for losing a life.")
                fitness_current -= 25
                lives_max = lives
            else:
                if lives > lives_max:
                    print("Added 25 Fitness for losing a life.")
                    fitness_current += 25
                    lives_max = lives
                
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if done or counter == 300:
                if fitness_current == 0:
                    print("Subtracted 10 Fitness for not making much progress.")
                    fitness_current -= 10
                    current_max_fitness = fitness_current
                done = True
                print("\nGenome " + str(genome_id) + "\nFitness - " + str(fitness_current) + "\n")

            genome.fitness = fitness_current
            
    
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "config-feedforward")

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restorefile)
else:
    p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

winner = p.run(eval_genomes)

with open("winner.pkl", "wb") as output:
    pickle.dump(winner, output, 1)
