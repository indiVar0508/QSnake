import pygame
import numpy as np
from collections import deque
import tensorflow as tf
import random
from PIL import ImageGrab
import cv2
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)

class Food:

    def __init__(self, size = 3, color = (0, 255, 0),location = (50, 50)):
        self.size = size
        self.color = color
        self.location = location

    def showFood(self, game =  None):
        if game == None:
            raise 'No Game to display.!'
            quit()
        pygame.draw.circle(game, self.color, self.location, self.size)
        

class snake:
    
    def __init__(self, gameWidth, gameHeight, foodCords, foodSize, length = 5, size = 5, color = (0, 255, 255), x = 0, y = 0, visionLimit = 50, showVision = False,\
     learningRate = 0.15, epsilon = 1.0, min_epsilon = 0.05, epsilon_decay = 0.995, gamma = 0.9):
        self.length = length
        self.size = size
        self.color = color
        self.headX = x
        self.headY = y
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.snakeCords = []
        for i in range(self.length): self.snakeCords.insert(0, (self.headX, self.headY - 2*i*self.size))
        self.headY -= 2*(self.length-1)*self.size
        self.speed = self.size
        self.dead = False
        self.left = self.right = self.up = self.down = False
        self.visionLimit = visionLimit
        self.showVision = showVision
        self.makeVision(gameWidth, gameHeight, foodCords, foodSize)
        self.memory = deque(maxlen = 2000)
        self.brain = self.makeBrain(learningRate)

    def reset(self, gameHeight, gameWidth, length, foodCords, foodSize):
    	self.headY =  np.random.randint(low = 50, high = gameHeight-50)
    	self.headX =  np.random.randint(low = 50, high = gameWidth-50)
    	self.length = length
    	self.snakeCords = []
    	for i in range(self.length): self.snakeCords.insert(0, (self.headX, self.headY - 2*i*self.size))
    	self.headY -= 2*(self.length-1)*self.size
    	self.dead = False
    	self.left = self.right = self.up = self.down = False
    	self.color = (0, 255, 255)
    	self.makeVision(gameWidth, gameHeight, foodCords, foodSize)

    def remember(self, state, action, reward, next_state, done):
    	self.memory.append((state, action, reward, next_state, done))

    def makeBrain(self, learningRate):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv2D(filters = 64, padding = 'same', kernel_size = 5, activation = 'relu', input_shape = [300, 300, 3]))
        model.add(tf.keras.layers.MaxPool2D(padding = 'valid', pool_size = 2, strides = 2))
        model.add(tf.keras.layers.Conv2D(filters = 32, padding = 'same', kernel_size = 5, activation = 'relu'))
        model.add(tf.keras.layers.MaxPool2D(padding = 'valid', pool_size = 2, strides = 2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
        model.add(tf.keras.layers.Dense(units = 64, activation = 'relu'))
        model.add(tf.keras.layers.Dense(units = 3, activation = 'softmax'))
        model.compile(loss = 'mse', optimizer = tf.keras.optimizers.Adam(lr = learningRate))
        return model

    def act(self, state):
    	if np.random.random() < self.epsilon: return np.random.randint(3)
    	actions_prob = self.brain.predict(state)
    	return np.argmax(actions_prob[0])

    def replay(self, batch_size):
    	minbatch = random.sample(self.memory, batch_size)
    	for state, action, reward, next_state, done in minbatch:
    		target = reward
    		if not done:
    			target = (reward + self.gamma * np.amax(self.brain.predict(next_state)[0]))
    		target_f = self.brain.predict(state)
    		target_f[0][action] = target
    		self.brain.fit(state, target_f, epochs = 1, verbose = 0)
    	if self.epsilon > self.min_epsilon:
    		self.epsilon *= self.epsilon_decay

    def showPlayerVision(self, gameDisplay = None):
    	if not self.showVision or gameDisplay == None: return
    	pygame.draw.line(gameDisplay, self.leftVision, (self.headX, self.headY), (self.leftBlock))
    	pygame.draw.line(gameDisplay, self.rightVision, (self.headX, self.headY), (self.rightBlock))
    	pygame.draw.line(gameDisplay, self.upVision, (self.headX, self.headY), (self.upBlock))
    	pygame.draw.line(gameDisplay, self.downVision, (self.headX, self.headY), (self.downBlock))
    	# pygame.draw.line(gameDisplay, self.leftTopVision, (self.headX, self.headY), (self.leftTopBlock))
    	# pygame.draw.line(gameDisplay, self.rightTopVision, (self.headX, self.headY), (self.rightTopBlock))
    	# pygame.draw.line(gameDisplay, self.leftBottomVision, (self.headX, self.headY), (self.leftBottomBlock))
    	# pygame.draw.line(gameDisplay, self.rightBottomVision, (self.headX, self.headY), (self.rightBottomBlock))    	

    def updatePosition(self):
        if len(self.snakeCords) > 1: 
            if self.snakeCords[1] == (self.headX,self.headY): 
                self.dead = True 
                self.color = (255, 0, 0)
        self.snakeCords.insert(0, (self.headX,self.headY))
        if len(self.snakeCords) > self.length: del self.snakeCords[-1]

    def eatenItSelf(self):
        for pos in self.snakeCords[1:]:
            if pos == self.snakeCords[0]:
                self.dead = True
                self.color = (255, 0, 0)
        return False

    def moveLeft(self): self.headX -= self.speed
    def moveRight(self): self.headX += self.speed
    def moveDown(self): self.headY += self.speed
    def moveUp(self): self.headY -= self.speed

    def makeVision(self, gameWidth, gameHeight, foodCords, foodSize):

    	self.leftBlock = (self.headX - self.visionLimit, self.headY)
    	self.rightBlock = (self.headX + self.visionLimit, self.headY)
    	self.upBlock = (self.headX, self.headY - self.visionLimit)
    	self.downBlock = (self.headX, self.headY + self.visionLimit)
    	# self.leftTopBlock = (self.headX - self.visionLimit, self.headY - self.visionLimit)
    	# self.rightTopBlock = (self.headX + self.visionLimit, self.headY - self.visionLimit)
    	# self.leftBottomBlock = (self.headX - self.visionLimit, self.headY + self.visionLimit)
    	# self.rightBottomBlock = (self.headX + self.visionLimit, self.headY + self.visionLimit)
    	self.leftDistance = self.rightDistance = self.upDistance = self.downDistance = self.visionLimit
    	# self.leftTopDistance = self.rightTopDistance = self.leftBottomDistance = self.rightBottomDistance = self.visionLimit
    	self.leftDetection = self.rightDetection = self.upDetection = self.downDetection = 0
    	# self.leftTopDetection = self.rightTopDetection = self.leftBottomDetection = self.rightBottomDetection = 0
    	self.leftVision = self.rightVision = self.upVision = self.downVision = (255, 255, 255)
    	# self.leftTopVision = self.rightTopVision = self.leftBottomVision = self.rightBottomVision = (255, 255, 255)
    	self.handleVision(gameWidth, gameHeight, foodCords, foodSize)

    def handleVision(self, gameWidth, gameHeight, foodCords, foodSize):
    	if self.headX - self.visionLimit < 0: 
    		self.leftBlock = (0, self.headY)
    		self.leftVision = (255, 0, 0)
    		self.leftDistance = self.headX
    		self.leftDetection = -1
    		# self.leftTopBlock = (0, self.headY - self.headX)
    		# self.leftBottomBlock = (0, self.headY + self.headX)
    		# self.leftTopVision = self.leftBottomVision = (255, 0, 0)
    		# self.leftTopDistance = self.leftBottomDistance = self.headX
    		# self.leftTopDetection = self.leftBottomDetection = -1
    	if self.headX + self.visionLimit > gameWidth:
    		self.rightBlock = (gameWidth, self.headY)
    		self.rightVision = (255, 0, 0)
    		self.rightDistance = (gameWidth - self.headX)
    		self.rightDetection = -1
    		# self.rightTopBlock = (gameWidth, self.headY - (gameWidth - self.headX))
    		# self.rightBottomBlock = (gameWidth, self.headY + gameWidth - self.headX)
    		# self.rightTopVision = self.rightBottomVision = (255, 0, 0)
    		# self.rightTopDistance = self.rightBottomDistance = (gameWidth - self.headX)
    		# sself.rightTopDetection = self.rightBottomDetection = -1
    	if self.headY - self.visionLimit < 0:
    		self.upBlock = (self.headX, 0)
    		self.upVision = (255, 0, 0)
    		self.upDistance = self.headY
    		self.upDetection = -1
    		# self.leftTopBlock = (self.headX - self.headY, 0)
    		# self.rightTopBlock = (self.headX + self.headY, 0)
    		# self.rightTopVision = self.upVision = (255, 0, 0)
    		# self.leftTopDistance = self.rightTopDistance = self.headY
    		# self.leftTopDetection = self.rightTopDetection = -1
    	if self.headY + self.visionLimit > gameHeight:
    		self.downBlock = (self.headX, gameHeight)
    		self.downVision = (255, 0, 0)
    		self.downDistance = (gameHeight - self.headY)
    		self.downDetection = -1
    		# self.leftBottomVision = (self.headX - self.headY, gameHeight)
    		# self.rightBottomVision = (self.headX + self.headY, gameHeight)
    		# self.rightBottomVision = self.downVision = (255, 0, 0)
    		# self.leftBottomDistance = self.rightBottomDistance = (gameHeight - self.headY)
    		# self.leftBottomDetection = self.rightBottomDetection = -1
    	self.foodDetected(foodCords, foodSize)

    def foodDetected(self, foodCords, foodSize):
    	if foodCords[0] - foodSize < self.headX < foodCords[0] + foodSize and self.headY - self.visionLimit < foodCords[1] < self.headY: 
            self.upDetection = 1
            self.upVision = (0, 0, 255)
    	elif foodCords[0] - foodSize < self.headX < foodCords[0] + foodSize and self.headY < foodCords[1] < self.headY + self.visionLimit: 
            self.downDetection = 1
            self.downVision = (0, 0 , 255)
    	elif foodCords[1] - foodSize < self.headY < foodCords[1] + foodSize and self.headX - self.visionLimit < foodCords[0] < self.headX: 
            self.leftDetection = 1
            self.leftVision = (0, 0, 255)
    	elif foodCords[1] - foodSize < self.headY < foodCords[1] + foodSize and self.headX < foodCords[0] < self.headX + self.visionLimit: 
            self.rightDetection = 1
            self.rightVision = (0, 0, 255)
    	# elif foodCords[0] - foodSize < self.headX - self.visionLimit < foodCords[0] + foodSize and self.headY - self.visionLimit < foodCords[1] < self.headY: 
     #        self.leftTopDetection = 1
     #        self.leftTopVision = (0, 0, 255)
    	# elif foodCords[0] - foodSize < self.headX - self.visionLimit < foodCords[0] + foodSize and self.headY < foodCords[1] < self.headY + self.visionLimit:
     #        self.leftBottomDetection = 1
     #        self.leftBottomVision = (0, 0, 255)
		# elif foodCords[0] - foodSize < self.headX + self.visionLimit < foodCords[0] + foodSize and self.headY - self.visionLimit < foodCords[1] < self.headY:
		# 	self.rightTopDetection = 1
		# 	self.rightTopVision = (0, 0, 255)
		# elif foodCords[0] - foodSize < self.headX + self.visionLimit < foodCords[0] + foodSize and self.headY < foodCords[1] < self.headY + self.visionLimit:
		# 	self.rightBottomDetection = 1
		# 	self.rightBottomVision = (0, 0 ,255)



    def getState(self):
    	# print(self.leftDistance / self.visionLimit, self.rightDistance / self.visionLimit, self.upDistance / self.visionLimit, self.downDistance / self.visionLimit)
    	# print(self.leftDetection, self.rightDetection,self.upDetection, self.downDetection)
    	return np.array((self.leftDistance / self.visionLimit, self.rightDistance / self.visionLimit, self.upDistance / self.visionLimit, self.downDistance / self.visionLimit,\
    			self.leftDetection, self.rightDetection,self.upDetection, self.downDetection)).reshape(1, -1)






    def showSnake(self, game = None, color = (255, 255, 255)):
        if game == None:
            raise 'No Game to display.!'
            quit()
        for pos in self.snakeCords: pygame.draw.circle(game, self.color, pos, self.size)

        

class game:

    def __init__(self, gameWidth = 200, gameHeight = 200):
        pygame.init()
        self.gameWidth = gameWidth
        self.gameHeight = gameHeight
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BACKGROUND = (51, 51, 51)

        self.food = Food(location = (np.random.randint(50, self.gameWidth- 50), np.random.randint(50, self.gameHeight - 50)))
        self.player = snake(gameWidth = gameWidth, gameHeight = gameHeight, foodCords = self.food.location, foodSize = self.food.size, x = np.random.randint(low = 50, high = self.gameWidth-50), y = np.random.randint(low = 50, high = self.gameHeight - 50), showVision = True)
        self.display = pygame.display.set_mode((self.gameWidth, self.gameHeight))
        pygame.display.set_caption('Snake')
        self.fps = pygame.time.Clock()
        self.score = 0
        self.frameRate = 50
        self.bestScore = 0
        

    def makeobjMsg(self, msg, fontD,color = (0, 0, 0)):
        return fontD.render(msg, True, color), fontD.render(msg, True, color).get_rect()
        
    def message(self, msg, color = (0, 0, 0), fontType = 'freesansbold.ttf', fontSize = 15, xpos = 10, ypos = 10):
        fontDefination = pygame.font.Font(fontType, fontSize)
        msgSurface, msgRectangle = self.makeobjMsg(msg, fontDefination, color)
        msgRectangle = (xpos, ypos)
        self.display.blit(msgSurface, msgRectangle)

    def pauseGame(self):

        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_r:
                        return
            self.message(msg = 'PAUSED.!',color = self.WHITE, fontSize = 30, xpos = self.gameWidth // 2 - 50, ypos = self.gameHeight // 2)
            pygame.display.update()

    def learn(self):
    	e = 0
    	while True:
    		self.playGame(e)
    		self.reset()
    		e += 1

    def reset(self):
    	self.moveFood()
    	self.score = 0
    	self.player.reset(self.gameHeight, self.gameWidth, 5, self.food.location, self.food.size)

    def playGame(self, e):

        # state = self.player.getState()
        state = self.readScreen()
        # print(state.shape)
        # quit()
        while not self.player.dead:
            
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        self.pauseGame()
                    if event.key == pygame.K_s:
                        self.player.showVision = not(self.player.showVision)

                    # if event.key == pygame.K_DOWN: 
                    #     self.player.down = True
                    #     self.player.up = self.player.left = self.player.right = False
                    # elif event.key == pygame.K_UP: 
                    #     self.player.up = True
                    #     self.player.down = self.player.left = self.player.right = False
                    # elif event.key == pygame.K_LEFT: 
                    #     self.player.left = True
                    #     self.player.up = self.player.down = self.player.right = False
                    # elif event.key == pygame.K_RIGHT:
                    #     self.player.right = True
                    #     self.player.up = self.player.down = self.player.left = False

                '''if event.type == pygame.KEYUP:
                    if event.key == pygame.K_DOWN: down = False
                    elif event.key == pygame.K_UP: up = False
                    elif event.key == pygame.K_LEFT: left = False
                    elif event.key == pygame.K_RIGHT: right = False'''
            action = self.player.act(state)
            reward = 0
            if action == 0:
            	reward += self.moveLeft()
            	self.player.left = True
            	self.player.up = self.player.down = self.player.right = False
            elif action == 1: 
            	reward += self.moveRight()
            	self.player.right = True
            	self.player.up = self.player.down = self.player.left = False
            elif action == 2: 
            	reward += self.moveUp()
            	self.player.up = True
            	self.player.down = self.player.left = self.player.right = False
            # elif action == 3: 
            # 	reward += self.moveDown()
            # 	self.player.down = True
            # 	self.player.up = self.player.left = self.player.right = False

            # next_state = self.player.getState()
            next_state = self.readScreen()

            self.player.makeVision(self.gameWidth, self.gameHeight, self.food.location, self.food.size)
            self.display.fill(self.BACKGROUND)
            self.showGame()
            done = self.foodEaten()
            if done:
                self.player.length += 1
                self.moveFood()
                self.score += 1
                reward = 5
            # self.message('Score = {} episode = {} bestScore = {}'.format(self.score, e, self.bestScore), color = self.WHITE)
            pygame.display.update()
            self.fps.tick(self.frameRate)
            self.player.remember(state, action, reward, next_state, done)
            if self.score > self.bestScore: self.bestScore = self.score
            # self.readScreen()
        if len(self.player.memory) > 64:
        	self.player.replay(64)

    def readScreen(self):
        screen = np.array(ImageGrab.grab(bbox = (0, 0, 1.5 * self.gameWidth, 1.5 * self.gameHeight)))
        return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB).reshape(-1, screen.shape[0], screen.shape[1], screen.shape[2])
        # print(screen.shape)
        # cv2.imshow('Stream', cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        

    def moveFood(self): self.food.location = (np.random.randint(low = 50, high = self.gameWidth-50), np.random.randint(low = 50, high = self.gameHeight-50))
    
    def foodEaten(self):
        fx, fy = self.food.location
        sx, sy = self.player.headX, self.player.headY
        if (sx - self.player.size < fx - self.food.size < sx + self.player.size or sx - self.player.size < fx + self.food.size < sx + self.player.size)\
           and (sy - self.player.size < fy-self.food.size < sy + self.player.size or sy - self.player.size < fy+self.food.size < sy + self.player.size): return True
        return False
    
    def showGame(self):
        self.player.showSnake(self.display,self.BLACK)
        self.player.showPlayerVision(self.display)
        self.food.showFood(self.display)
        
    def moveLeft(self):
        self.player.moveLeft()
        if self.player.headX - self.player.size < 0: 
            self.player.headX = self.player.size 
            self.player.dead = True #self.player.headX = self.gameWidth - self.player.size
            self.player.color = (255, 0, 0)
            return -1
        self.player.updatePosition()
        self.player.eatenItSelf()
        return 0.5

    def moveRight(self):
        self.player.moveRight()
        if self.player.headX + self.player.size > self.gameWidth:
            self.player.headX = self.gameWidth - self.player.size 
            self.player.dead = True #self.player.headX = self.player.size
            self.player.color = (255, 0, 0)
            return -1
        self.player.updatePosition()
        self.player.eatenItSelf()
        return 0.5

    def moveDown(self):
        self.player.moveDown()
        if self.player.headY + self.player.size > self.gameHeight: 
            self.player.headY = self.gameHeight - self.player.size 
            self.player.dead = True #self.player.headY = self.player.size
            self.player.color = (255, 0, 0)
            return -1
        self.player.updatePosition()
        self.player.eatenItSelf()
        return 0.5

    def moveUp(self):
        self.player.moveUp()
        if self.player.headY - self.player.size < 0:
            self.player.headY = self.player.size 
            self.player.dead = True #self.gameHeight - self.player.size
            self.player.color = (255, 0, 0)
            return -1
        self.player.updatePosition()
        self.player.eatenItSelf()
        return 0.5



if __name__ == '__main__':
    g = game()
    g.pauseGame()
    g.learn()
    pygame.quit()      
