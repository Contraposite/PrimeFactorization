#!/usr/bin/env python
# coding: utf-8

# In[1]:


from kivy.config import Config
Config.set('graphics', 'fullscreen', '0')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.progressbar import ProgressBar
from kivy.uix.gridlayout import GridLayout
from kivy.properties import NumericProperty, ObjectProperty, ListProperty
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.lang import Builder


# In[2]:


import random


# In[3]:


def get_factors(n):
    prime_factors = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            prime_factors.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
        prime_factors.append(n)
    return prime_factors


# In[4]:


class MyWindowManager(ScreenManager):
    pass


# In[5]:


class TextLabel(Label):
    pass


# In[6]:


class MultilineTextLabel(Label):
    pass


# In[7]:


keycodes = {
    49:'1',
    257:'1',
    50:'2',
    258:'2',
    51:'3',
    259:'3',
    52:'4',
    260:'4',
    53:'5',
    261:'5',
    54:'6',
    262:'6',
    55:'7',
    263:'7',
    56:'8',
    264:'8',
    57:'9',
    265:'9',
    48:'0',
    256:'0',
    
}


# In[8]:


class MyGrid(Screen):
    target = ObjectProperty(None)
    factors = ObjectProperty(None)
    running_input = ObjectProperty(None)
    time_display = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()
        
    def reset(self):
        self.score = 0
        self.question_time = 100
        self.time_remaining = 100
        self.game_time = 0
        self.new_challenge(1)
    
    def new_challenge(self, game_time):
        self.answer = ""
        self.running_input.text = ""
        self.question_time = 5 + 10/(0.1+(game_time**0.05))
        self.time_remaining = self.question_time
        self.target_number = random.randint(int(2+2*game_time),int(11+2*game_time))
        self.target.text = str(self.target_number)
        self.factors_so_far = []
        self.factors.text = str(self.factors_so_far)
        self.factors_to_go = get_factors(self.target_number)
        
        
    def submit(self, key=""):
        self.answer = self.answer + key
        
        submission_text = self.answer
        submission = int(submission_text)

        if self.factors_to_go[0] == submission:
            self.factors_to_go.remove(submission)
            self.factors_so_far.append(submission)
            self.factors.text = str(self.factors_so_far)
            self.answer = ""
        elif submission_text in str(self.factors_to_go[0]) and str(self.factors_to_go[0]).index(submission_text)==0:
            pass #this is part way towards the correct answer. let them continue writing.
        else:
            self.lose_game(submission_text)
        
        if len(self.factors_to_go) < 1:
            self.new_challenge(self.game_time)
            self.score += 1
            
        self.running_input.text = self.answer
    
    def key_press(self, keyboard, keycode, text, modifiers, other):
        if keycode in keycodes:
            key = keycodes[keycode]
            self.submit(key)
    
    def update(self, dt):
        if self.manager.current == 'game_screen':
            self.game_time += dt
            self.time_remaining -= dt
            self.time_display.value = self.time_remaining / self.question_time
            if self.time_remaining <0:
                self.lose_game()

    def lose_game(self, losing_attempt='None'):
        self.manager.current = 'game_over_screen'
        self.manager.screens[1].info_to_display.text = "score: " + str(self.score) + "\ntarget: " + str(self.target_number) + "\nfactors so far: " + self.factors.text + "\ninput: " + losing_attempt + "\ncorrect answer: \n" + str(get_factors(self.target_number))


# In[9]:


class GameOver(Screen):
    
    info_to_display = ObjectProperty(None)
    
    def restart(self):
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[10]:


class PrimeFactorizerApp(App):
    def build(self):
        root = Builder.load_file('PrimeFactorizer_UI.kv')
        
        SM = MyWindowManager()
        
        my_grid = MyGrid()
        game_over = GameOver()
        Clock.schedule_interval(my_grid.update, 1.0/60.0)
        
        SM.add_widget(my_grid)
        SM.add_widget(game_over)
        
        Window.bind(on_key_down = my_grid.key_press)
        
        return SM

    
if __name__ == '__main__':
    PrimeFactorizerApp().run()


# In[ ]:




