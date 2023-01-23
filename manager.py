import pygame
import random
import pandas as pd
import numpy as np
import time
from random import randint, sample
from point import Point
from utils import *
from genetic import Genetic
from ant_colony import *
from ant import *
# from main import num_of_points

offset          = 100
width, height   = 1920, 1080
populationSize  = 300
n = 7
colony_size = 10
iterations = 300
pygame.font.init()

time_df=pd.read_csv('time.csv')

class Manager(object):
    size            = (width, height)
    fps             = 30
    screen          = pygame.display.set_mode(size)
    clock           = pygame.time.Clock()
    scaler          = 1
    max_radius      = 15
    Black           = (0, 0, 0)
    White           = (255, 255, 255)
    Yellow          = (156, 203, 190)
    Gray            = (100, 100, 100)
    Highlight       = (156, 203, 190)
    LineThickness   = 4
    showIndex       = True
    n_points        = n
    algorithms      = ["Brute Force", "Lexicographic Order", "Genetic Algorithm", "Ant Colony ACS", "Ant Colony Elitist", "Ant Colony Max-Min"]

    genetic         = Genetic([sample(list(range(n)), n) for i in range(populationSize)], populationSize)

    PossibleCombinations = Factorial(n_points)
    # print("possible combinations : {}".format(Factorial(n_points)))

    Order           = [i for i in range(n_points)]
    counter         = 0
    start_time=0
    time=0
    stop=False
    percent=0

    def __init__(self, Points = [Point(randint(offset, width-offset), randint(offset, height-offset)) for i in range(n_points)]):
        self.Points          = Points
        self.recordDistance  = SumDistance(self.Points)
        self.OptimalRoutes   = self.Points.copy()
        self.currentList     = self.Points.copy()

        # --- Ant Colony ---
        self.antColony = AntColony(variation="ACS", size=colony_size, max_iterations = iterations,
                         nodes=self.Points.copy(), alpha=1, beta=3, rho=0.1, pheromone=1, phe_deposit_weight=1)

    def SetNumberOfPoints(self, num):
        self.n_points=num

    def ResetGenetic(self):
        self.genetic = Genetic([sample(list(range(n)), n) for i in range(populationSize)], populationSize)

    def ChangeAntColonyVariation(self, name):
        self.antColony.variation = name

    def ResetAntColony(self, name="ACS"):
        self.recordDistance  = SumDistance(self.Points)
        self.antColony = AntColony(variation=name, size=colony_size, max_iterations = iterations,
                         nodes=self.Points.copy(), alpha=1, beta=3, rho=0.1, pheromone=1, phe_deposit_weight=1)
    def SetFps(self):
        return self.clock.tick(self.fps)/1000.0

    def UpdateCaption(self):
        frameRate = int(self.clock.get_fps())
        pygame.display.set_caption("Traveling Salesman Problem - Fps : {}".format(frameRate))

    def Counter(self):
        self.counter += 1
        if self.counter > self.PossibleCombinations:
            self.counter = self.PossibleCombinations

    def ResetTime(self):
        self.start_time=time.time()

    def Time(self):
        if not self.stop:
            self.time=time.time()-self.start_time
        self.ShowTime()
            
    def ShowTime(self):
        textColor   = (255, 255, 255)
        # textFont    = pg.font.Font("freesansbold.ttf", size)
        textFont    = pygame.font.SysFont("Arial", 20)
        textSurface = textFont.render(str(round(self.time, 4)), False, textColor)
        self.screen.blit(textSurface, (width//2, 50))

    def WriteTimeData(self, method):
        df=pd.read_csv('time.csv')
        index=df.index[(df['Method']==method) & (df['NumberOfPoints']==self.n_points)][0]
        df.loc[index, ['Time']]=self.time
        df=df[['Method', 'NumberOfPoints', 'Time']]
        df.to_csv('time.csv')

    def BruteForce(self):
        if self.counter != self.PossibleCombinations:
            i1 = randint(0, self.n_points-1)
            i2 = randint(0, self.n_points-1)
            self.Points[i1], self.Points[i2] = self.Points[i2], self.Points[i1]
        else:
            self.stop=True
            self.WriteTimeData('BruteForce')

        self.Counter()

        dist = SumDistance(self.Points)
        if dist < self.recordDistance:
            self.recordDistance  = dist
            self.OptimalRoutes   = self.Points.copy()
            # print("Shortest distance : {}" .format(self.recordDistance))

        self.DrawLines()

    def Lexicographic(self):
        x=0
        if x!=-1:
            self.Order, x = LexicalOrder(self.Order)
            nodes = []
            for i in self.Order:
                nodes.append(self.Points[i])
        else:
            self.stop=True
            self.WriteTimeData('LexicographicOrder')

        self.Counter()

        dist = SumDistance(nodes)
        if dist < self.recordDistance:
            self.recordDistance = dist
            self.OptimalRoutes  = nodes.copy()
            #print("Shortest distance : {}" .format(self.recordDistance))
        self.DrawLines()

    def GeneticAlgorithm(self):
        if np.square(np.sum(np.power(self.genetic.fitness)))<=0.8:
            self.genetic.CalculateFitness(self.Points)
            self.genetic.NaturalSelection()
        else:
            self.stop=True
            self.WriteTimeData('GeneticAlgorithm')

        # self.Counter()
        for i in range(self.n_points):
            self.currentList[i] = self.Points[self.genetic.current[i]]
        if self.genetic.record < self.recordDistance:
            for i in range(self.n_points):
                self.OptimalRoutes[i] = self.Points[self.genetic.fitest[i]]
            self.recordDistance = self.genetic.record
        # print(self.OptimalRoutes)

        self.DrawLines(True)

    def AntColonyOptimization(self, pause):
        if pause == False:
            self.counter += 1
            if self.counter > self.antColony.max_iterations:
                self.counter = self.antColony.max_iterations

            if self.counter < self.antColony.max_iterations:
                self.antColony.Simulate(self.counter)
        if self.percent==100.0:
            self.stop=True
            if self.antColony.variation=='ACS':
                variation='AntColonyACS'
            elif self.antColony.variation=='ELITIST':
                variation='AntColonyElitist'
            else:
                variation='AntColonyMaxMin'
            self.WriteTimeData(variation)

        self.antColony.Draw(self)
        self.recordDistance = self.antColony.best_distance

    def RandomPoints(self):
        self.Points = [Point(randint(offset, width-offset), randint(offset, height-offset)) for i in range(self.n_points)]
        self.ResetAntColony(self.antColony.variation)
        self.recordDistance  = SumDistance(self.Points)
        self.OptimalRoutes   = self.Points.copy()
        self.currentList     = self.Points.copy()

    def Percentage(self, val):
        self.percent = (self.counter/val) * 100
        # textColor   = (255, 255, 255)
        # # textFont    = pg.font.Font("freesansbold.ttf", size)
        # textFont    = pygame.font.SysFont("Arial", 20)
        # textSurface = textFont.render(str(round(self.percent, 4)), False, textColor)
        # self.screen.blit(textSurface, (width//2, 50))


    def ShowText(self, selectedIndex, started = True):
        textColor   = (255, 255, 255)
        # textFont    = pg.font.Font("freesansbold.ttf", size)
        textFont    = pygame.font.SysFont("Times", 20)
        textFont2    = pygame.font.SysFont("Arial Black", 40)

        textSurface1 = textFont.render("Best distance : " + str(round(self.recordDistance,2)), False, textColor)
        textSurface2 = textFont.render(self.algorithms[selectedIndex], False, textColor)
        textSurface3 = textFont2.render("... Press ' SPACE ' to start ..." ,False, textColor)

        self.screen.blit(textSurface1, (100, 70))
        self.screen.blit(textSurface2, (100, 35))
        if started == False:
            self.screen.blit(textSurface3, (width//2, height-200))

    def DrawShortestPath(self):
        if len(self.OptimalRoutes) > 0:
            for n in range(self.n_points):
                _i = (n+1)%self.n_points
                pygame.draw.line(self.screen, self.Highlight,
                                (self.OptimalRoutes[n].x, self.OptimalRoutes[n].y),
                                (self.OptimalRoutes[_i].x, self.OptimalRoutes[_i].y),
                                self.LineThickness)
                self.OptimalRoutes[n].Draw(self, self.showIndex, True, n)

    def DrawPoints(self, selected_index = 0):
        for point in self.Points:
            point.radius = self.scaler
            point.Draw(self)

    def DrawLines(self, drawCurrent=False):
        if drawCurrent == True:
            for i, point in enumerate(self.currentList):
                _i = (i+1)%len(self.currentList)
                pygame.draw.line(self.screen, self.Gray, (point.x, point.y), (self.currentList[_i].x, self.currentList[_i].y), 1)
        else:
            for i, point in enumerate(self.Points):
                _i = (i+1)%len(self.Points)
                pygame.draw.line(self.screen, self.Gray, (point.x, point.y), (self.Points[_i].x, self.Points[_i].y), 1)

    def Background(self):
        self.screen.fill(self.Black)