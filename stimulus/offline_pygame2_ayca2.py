#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
"""

import pygame, random, time, os, sys
import numpy as np

#we will borrow PsychoPy's parallel feature to record stimulus timings using the parallel port
# from psychopy import core, parallel
#parallel.setPortAddress(61432) #61432 is the lab port address

def offline():

    pygame.init() #start pygame
    start_time = time.time()
    time_past_total = time.time() - start_time
    print(f"time passed: {time_past_total}")

    #specify screen and background sizes
    screen = pygame.display.set_mode((600,600))
    screenrect = screen.get_rect()
    background = pygame.Surface((screen.get_size()))
    backgroundrect = background.get_rect()
    background.fill((0,0,0)) # black
    background = background.convert()
    screen.blit(background, (0,0)) # clean whole screen

    clock = pygame.time.Clock()
    mainloop = True
    FPS = 2 # 2 FPS should give us epochs of 500 ms


    #specify the grid content

    # grid = [
    #
    #         "<^", ">v",
    # ]

    grid = [
        " ^ ",
        "< >",
        " v ",
    ]

    phrase = "abcd" #this is used to store the string at the bottom of the interface

    lines = len(grid)
    columns = len(grid[0])
    print(f"num of lines: {lines} and num of columns: {columns}")
    length = screenrect.width / columns
    height = screenrect.height / lines
    print(f"length: {length} and height: {height}")

    oldhighlight = [0,0]

    numtrials = 0 # counter for highlighting, when it is equal to n_trials, new target is made
    num_iterations = 3
    # targets = [[1,1],[3,5],[1,0],[2,2],[3,1],[4,0],[6,5]]
    targets = [[0,1],[1,0],[1,2],[2,1]]
    targets_list = targets*num_iterations
    n_trials = len(targets_list)
    print("AAAAAAAAAAAAAAA", n_trials)

    targetcounter = 0

    waittime = 3000#3000

    #we will use this function to write the letters
    def write(msg, colour=(30,144,255)):
        myfont = pygame.font.SysFont("None", 190)
        mytext = myfont.render(msg, True, colour)
        mytext = mytext.convert_alpha()
        return mytext

    #use this function to write the spelled phrase
    def writePhrase():
        for z in range(len(phrase)):
            textsurface = write(phrase[z], (255,255,255))
            background.blit(textsurface, (length * z + length/4, height * 5 + height/4))

    #generate uncoloured frame
    def makeStandard():
        for y in range(lines):
            for x in range(columns):
                textsurface = write(grid[y][x])
                background.blit(textsurface, (length * x + length/4, height * (y) + height/4))

        writePhrase()
        screen.blit(background, (0,0))
        pygame.display.flip()

    #this function makes a makes a target
    def makeTarget(target):
        for y in range(lines):
            for x in range(columns):
                if y == target[0] and x == target[1]:
                    textsurface = write(grid[y][x],(255,0,0))
                    background.blit(textsurface, (length * x + length/4, height * (y) + height/4))
                else:
                    textsurface = write(grid[y][x])
                    background.blit(textsurface, (length * x + length/4, height * (y) + height/4))
        writePhrase()

    def makeHighlighted(target_list, counter, oldhighlight):
        highlight = target_list[counter]
        print("GONNA LOSE MY MIND",oldhighlight)
        while (highlight[0] == oldhighlight[0]) and (highlight[1] == oldhighlight[1]) : #adjusts repeated values
            print("BBBBBBBBBBB", highlight)
            highlight = random.choice([[0, 1], [1, 0], [1, 2], [2, 1]])
            print("CCCCCCCCCCC", highlight)
        
        textsurface = write(grid[highlight[1]][highlight[0]],(255,255,100))
        background.blit(textsurface, (length * highlight[0] + length/4, height * (highlight[1]) + height/4))     

        writePhrase()


        return highlight

    #pygame uses a main loop to generate the interface
    while mainloop:
        milliseconds = clock.tick(FPS)  # milliseconds passed since last frame
        seconds = milliseconds / 1000.0 # seconds passed since last frame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                mainloop = False # pygame window closed by user
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    mainloop = False # user pressed ESC

        if targetcounter < len(targets):
            if numtrials == 0:
                makeTarget(targets[targetcounter])
                screen.blit(background, (0,0)) #clean whole screen
                pygame.display.flip()
                pygame.time.wait(waittime)
                numtrials += 1
            elif numtrials == (n_trials):
                targetcounter += 1

                time_past_total = time.time() - start_time
                print(f"time passed, end of symbol session: {time_past_total}")
                numtrials = 0
            else:
                print(numtrials)
                makeStandard()
                pygame.time.wait(2)
                np.random.shuffle(targets_list)
                oldhighlight = makeHighlighted(targets_list, numtrials, oldhighlight)
                print(f"old highlight {oldhighlight}")

                screen.blit(background, (0,0)) # clean whole screen
                pygame.display.flip()
                # print(f" msec {clock.get_time()}")
                # print(f" raw msec {clock.get_rawtime()}")

                time_past_total = time.time() - start_time
                print(f"time passed: {time_past_total}")

                numtrials += 1

        else:
            pygame.quit()
    


#currently the scripts are written to be run as standalone
#routines. We should change these to work in conjunction once we get
#the classifiers working sometime in the future.
if __name__=="__main__":
    offline()