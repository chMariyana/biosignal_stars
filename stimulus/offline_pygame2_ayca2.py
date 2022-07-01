#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
"""

import pygame, random, time, os, sys
import numpy as np
from pylsl import local_clock, StreamInfo, StreamOutlet

#
# we will borrow PsychoPy's parallel feature to record stimulus timings using the parallel port
# from psychopy import core, parallel
# parallel.setPortAddress(61432) #61432 is the lab port address

def offline():

    pygame.init() #start pygame
    start_time = time.time()
    time_past_total = time.time() - start_time
    print(f"time passed: {time_past_total}")

    # # Define the marker stream
    info = StreamInfo(name='pygame_markers',
                        type='Markers',
                        channel_count=1,
                        nominal_srate=0,
                        channel_format='string',
                        source_id='pygame_markers')
    marker_stream = StreamOutlet(info)

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
    grid = [
        " ^ ",
        "< >",
        " v ",]

    phrase = "abcd" #this is used to store the string at the bottom of the interface

    lines = len(grid)
    columns = len(grid[0])
    print(f"num of lines: {lines} and num of columns: {columns}")
    length = screenrect.width / columns
    height = screenrect.height / lines
    print(f"length: {length} and height: {height}")

    oldhighlight = [0,0]

    
    num_iterations = 3
    # targets = [[1,1],[3,5],[1,0],[2,2],[3,1],[4,0],[6,5]]
    targets = [[0,1],[1,0],[1,2],[2,1]]
    targets_list = targets*num_iterations
    n_repetitions = len(targets_list)
    n_trials = 4
    n_blocks = 5
    
    # counters
    repetition_no = 0 # counter for highlighting, when it is equal to n_repetitions, new target is made
    trial_no = 0
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
        while (highlight[0] == oldhighlight[0]) and (highlight[1] == oldhighlight[1]) : #adjusts repeated values
            highlight = random.choice([[0, 1], [1, 0], [1, 2], [2, 1]])
        
        textsurface = write(grid[highlight[1]][highlight[0]],(255,255,100))
        background.blit(textsurface, (length * highlight[0] + length/4, height * (highlight[1]) + height/4))     

        writePhrase()


        return highlight


    milliseconds = clock.tick(FPS)  # milliseconds passed since last frame
    seconds = milliseconds / 1000.0 # seconds passed since last frame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            mainloop = False # pygame window closed by user
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                mainloop = False # user pressed ESC
    for block_no in np.arange(n_blocks):
        print("block_no",block_no)
        if trial_no < n_trials:
            print(f"start trial: {trial_no}")
            if repetition_no == 0:
                print(f"start new target: {targets[repetition_no]}")

                makeTarget(targets[trial_no])
                screen.blit(background, (0,0)) #clean whole screen
                pygame.display.flip()
                pygame.time.wait(waittime)
                repetition_no += 1
            elif repetition_no == (n_repetitions):


                time_past_total = time.time() - start_time
                print(f"time passed, end of symbol session: {time_past_total}")
                repetition_no = 0
            else:
                makeStandard()
                pygame.time.wait(2)
                np.random.shuffle(targets_list)
                oldhighlight = makeHighlighted(targets_list, repetition_no, oldhighlight)
                print(f"old highlight {oldhighlight}")

                screen.blit(background, (0,0)) # clean whole screen
                pygame.display.flip()
                # print(f" msec {clock.get_time()}")
                # print(f" raw msec {clock.get_rawtime()}")

                time_past_total = time.time() - start_time
                print(f"time passed: {time_past_total}")

            trial_no += 1

        block_no += 1
        # pygame.quit()
        
    


#currently the scripts are written to be run as standalone
#routines. We should change these to work in conjunction once we get
#the classifiers working sometime in the future.
if __name__=="__main__":
    offline()