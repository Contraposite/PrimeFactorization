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
from kivy_garden.graph import LinePlot
from kivy_garden.graph import ScatterPlot


# In[2]:


import random
import pickle


# In[3]:


def save(obj):
    try:
        with open("save_data.pickle", "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('data saved!')
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)


# In[4]:


def load(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)


# In[5]:


#get/create the dictionary which will be the saved object storing important data
data = load('save_data.pickle')
if data == None:
    data = {'best_run': None, 'target_history': {'0':{'wins':0, 'losses':0, 'avg_time':0}}}
    
print(data)


# In[6]:


def record_win(target_number, time_taken, game_type):
    #ignore function call if the game type is training
    if game_type == 'training':
        return
    #make sure a dictionary exists for this target
    if not str(target_number) in data['target_history']:
        data['target_history'][str(target_number)] = {'wins':0, 'losses':0, 'avg_time':0}
    #modify values
    avg_before = data['target_history'][str(target_number)]['avg_time']
    wins_before = data['target_history'][str(target_number)]['wins']
    data['target_history'][str(target_number)]['wins'] += 1 #increment the number of wins for this target
    wins_after = data['target_history'][str(target_number)]['wins']
    data['target_history'][str(target_number)]['avg_time'] = (avg_before*wins_before + time_taken) / wins_after #set the average (finds the total time taken and divides by total wins)


# In[7]:


def record_loss(target_number, game_type):
    #ignore function call if the game type is training
    if game_type == 'training':
        return
    #make sure a dictionary exists for this target
    if not str(target_number) in data['target_history']:
        data['target_history'][str(target_number)] = {'wins':0, 'losses':0, 'avg_time':0}
    #modify values
    data['target_history'][str(target_number)]['losses'] += 1 #increment the number of losses for this target


# In[8]:


#this function is not mine
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


# In[9]:


class MyWindowManager(ScreenManager):
    pass


# In[10]:


class TextLabel(Label):
    pass


# In[11]:


class MultilineTextLabel(Label):
    pass


# In[12]:


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


# In[13]:


class MyGrid(Screen):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.game_type = 'normal'
        self.ids.mode_btn.text = 'norm'
        self.ids.mode_btn.color = (0.5,0.5,0.5,1)
        self.reset()
        
        self.TTS_plot = LinePlot( color=[0,0,1,1], line_width=2 )
        self.ids.TTS_graph.add_plot(self.TTS_plot)
        self.ids.TTS_graph.x_ticks_major=5
        self.TTS_plot.points = []
        
        Clock.schedule_interval(self.update_graphs, 1/10)
        
        self.training_set = [2] #just want it to not be empty as that would cause errors
    
    def update_graphs(self, dt):
        TTS_points_with_prediction = self.TTS_graph_points + [(self.score+1, self.question_time-self.time_remaining)]
        self.TTS_plot.points = TTS_points_with_prediction
        self.ids.TTS_graph.xmax = self.score+1
        self.ids.TTS_graph.ymax = self.longest_time_taken+1        
    
    def reset(self):
        self.score = 0
        self.question_time = 100
        self.time_remaining = 100
        self.game_time = 0
        self.TTS_graph_points = [(0,0)]
        self.longest_time_taken = 0
        self.ST_graph_points = [(0,0)]
        self.new_challenge(1)

    def new_challenge(self, game_time):
        self.answer = ""
        self.ids.running_input.text = ""
        self.question_time = 5 + 10/(0.1+(game_time**0.05))
        self.time_remaining = self.question_time
        self.target_number = self.new_target(game_time)
        self.ids.target.text = str(self.target_number)
        self.factors_so_far = []
        self.ids.factors.text = str(self.factors_so_far)
        self.factors_to_go = get_factors(self.target_number)
        
    def new_target(self, game_time):
        if self.game_type == 'normal':
            return random.randint(int(2+2*game_time),int(11+2*game_time))
        elif self.game_type == 'training':
            return random.choice(self.training_set)
        
    def submit(self, key=""):
                
        #if not on the game screen, don't submit anything
        if not self.manager.current == 'game_screen':
            return
        
        self.answer = self.answer + key
        
        submission_text = self.answer
        submission = int(submission_text)
        
        if self.factors_to_go[0] == submission:
            self.factors_to_go.remove(submission)
            self.factors_so_far.append(submission)
            self.ids.factors.text = str(self.factors_so_far)
            self.answer = ""
        elif submission_text in str(self.factors_to_go[0]) and str(self.factors_to_go[0]).index(submission_text)==0:
            pass #this is part way towards the correct answer. let them continue writing.
        else:
            record_loss(self.target_number, self.game_type)
            self.lose_game(submission_text)
        
        if len(self.factors_to_go) < 1: #if the player has listed all of the prime factors
            self.score += 1
            
            time_taken = self.question_time-self.time_remaining
            self.TTS_graph_points.append((self.score, time_taken))
            if time_taken > self.longest_time_taken:
                self.longest_time_taken = time_taken
            
            record_win(self.target_number, time_taken, self.game_type)
                
            self.ST_graph_points.append((self.game_time, self.score))
            
            self.new_challenge(self.game_time)
            
        self.ids.running_input.text = self.answer
            
    def key_press(self, keyboard, keycode, text, modifiers, other):
        if keycode in keycodes:
            key = keycodes[keycode]
            self.submit(key)
    
    def update(self, dt):
        if self.manager.current == 'game_screen':
            self.game_time += dt
            self.time_remaining -= dt
            self.ids.time_display.value = self.time_remaining / self.question_time
            if self.time_remaining <0:
                record_loss(self.target_number, self.game_type)
                self.lose_game()
    
    def cycle_mode(self):
        if self.game_type == 'normal':
            self.game_type = 'training'
            self.ids.mode_btn.text = 'train'
            self.ids.mode_btn.color = (1,0.9,0.0,1)
            #set training set
            confidence_ratings = [data['target_history'][key]['wins'] / (data['target_history'][key]['losses']+1) for key in data['target_history']]
            confidence_ratings.sort()
            threshold = confidence_ratings[int(len(confidence_ratings)*0.1)]
            self.training_set = [2]
            for key in data['target_history']:
                if data['target_history'][key]['wins'] / (data['target_history'][key]['losses']+1) <= threshold:
                    if key != '0':
                        self.training_set.append(int(key))
            self.reset()
        else:
            self.game_type = 'normal'
            self.ids.mode_btn.text = 'norm'
            self.ids.mode_btn.color = (0.5,0.5,0.5,1)
            self.reset()
    
    def lose_game(self, losing_attempt='None'):
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_over_screen'
        self.manager.screens[1].info_to_display.text = "score: " + str(self.score) + "\ntarget: " + str(self.target_number) + "\nfactors so far: " + self.ids.factors.text + "\ninput: " + losing_attempt + "\ncorrect answer: \n" + str(get_factors(self.target_number))
        self.manager.screens[1].update_info(self.game_type)
        
    def show_info(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'info_screen'
        
    def show_stats(self):
        self.manager.transition.direction = 'right'
        self.manager.screens[3].update_info()
        self.manager.current = 'stats_screen'
        


# In[14]:


class GameOver(Screen):
    
    info_to_display = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ids.ST_graph.y_grid=True
        self.ids.ST_graph.y_grid_label=True
        self.ids.ST_graph.y_ticks_major=5
        self.ids.ST_graph.x_grid=True
        self.ids.ST_graph.x_grid_label=True
        self.ids.ST_graph.x_ticks_major=10
        
        self.ST_plot_best = LinePlot( color=[1,0,0,1], line_width=1.5 )
        self.ids.ST_graph.add_plot(self.ST_plot_best)
        self.ST_plot_best.points = []
        
        self.ST_plot = LinePlot( color=[0,0,1,1], line_width=1.5 )
        self.ids.ST_graph.add_plot(self.ST_plot)
        self.ST_plot.points = []
        
    def update_info(self, game_type):
        self.ST_plot.points = self.manager.screens[0].ST_graph_points
        
        this_score = self.ST_plot.points[-1][1]
        this_run_time = self.ST_plot.points[-1][0]
        best_run_time = None #set this in the code below
        
        if game_type == 'normal':
            #check against saved high score
            if data['best_run'] == None:
                data['best_run'] = list(self.ST_plot.points)
                high_score = data['best_run'][-1][1]
                best_run_time = data['best_run'][-1][0]
            else:
                high_score = data['best_run'][-1][1]
                best_run_time = data['best_run'][-1][0]
                if this_score > high_score:
                    data['best_run'] = list(self.ST_plot.points)
                elif (this_score == high_score) and (this_run_time < best_run_time):
                    data['best_run'] = list(self.ST_plot.points)
            self.ST_plot_best.points = data['best_run']
        elif game_type == 'training':
            if data['best_run'] == None:
                self.ST_plot_best.points = [(0,0)]
                best_run_time = 0
                high_score = 0
            else:
                self.ST_plot_best.points = data['best_run']
                best_run_time = data['best_run'][-1][0]
                high_score = data['best_run'][-1][1]
            
        self.ids.ST_graph.xmax = max(this_run_time, best_run_time)+1
        self.ids.ST_graph.ymax = max(this_score, high_score)+1

        save(data)
        
    def restart(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[15]:


class Info(Screen):
        
    def restart(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[16]:


class Stats(Screen):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ids.WRN_graph.x_grid=True
        self.ids.WRN_graph.x_grid_label=True
        self.ids.WRN_graph.x_ticks_major=5
        self.ids.TN_graph.x_grid=True
        self.ids.TN_graph.x_grid_label=True
        self.ids.TN_graph.x_ticks_major=5
        
        self.WRN_plot = ScatterPlot( color=[0,0,1,1], point_size=2 )
        self.ids.WRN_graph.add_plot(self.WRN_plot)
        self.WRN_plot.points = []
        
        self.TN_plot = ScatterPlot( color=[0,0,1,1], point_size=2 )
        self.ids.TN_graph.add_plot(self.TN_plot)
        self.TN_plot.points = []
    
    def update_info(self):
        self.WRN_plot.points = []
        self.TN_plot.points = []
                
        #iterate through data and add points to graphs as we go
        for key in data['target_history'].keys():
            wins = data['target_history'][key]['wins']
            losses = data['target_history'][key]['losses']
            avg_time = data['target_history'][key]['avg_time']
            self.WRN_plot.points.append((int(key), wins/(losses+1)))
            if not avg_time == 0:
                self.TN_plot.points.append((int(key), avg_time))
        
        self.ids.WRN_graph.xmax = max(pt[0] for pt in self.WRN_plot.points)+1
        self.ids.WRN_graph.ymax = max(pt[1] for pt in self.WRN_plot.points)+1 +0.01*random.random() #random number fixes an issue with kivy only displaying the graph if it is different from before
        
        self.ids.TN_graph.xmax = max(pt[0] for pt in self.TN_plot.points)+1
        self.ids.TN_graph.ymax = max(pt[1] for pt in self.TN_plot.points)+1 +0.01*random.random() #random number fixes an issue with kivy only displaying the graph if it is different from before
                
    def restart(self):
        self.manager.transition.direction = 'left'
        self.manager.current = 'game_screen'
        self.manager.screens[0].reset()


# In[17]:


class PrimeFactorizerApp(App):
    def build(self):
        root = Builder.load_file('PrimeFactorizer_UI.kv')
        
        SM = MyWindowManager()
        
        my_grid = MyGrid()
        game_over = GameOver()
        info = Info()
        stats = Stats()
        Clock.schedule_interval(my_grid.update, 1.0/60.0)
        
        SM.add_widget(my_grid)
        SM.add_widget(game_over)
        SM.add_widget(info)
        SM.add_widget(stats)
        
        Window.bind(on_key_down = my_grid.key_press)
        
        return SM

    
if __name__ == '__main__':
    PrimeFactorizerApp().run()


# In[ ]:





# In[ ]:




