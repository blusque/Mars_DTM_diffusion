import sys
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
from typing import Callable, Tuple
import time


class LineDrawer:
  def __init__(self, fig, ax) -> None:
    self.ax = ax
    self.fig = fig
    self.xs = []
    self.ys = []
    self.line = None
    self.cid_1 = None
    self.cid_2 = None
    self.on = False
    self.mode = 'normal'
      
  def draw(self, *callbacks:Tuple[Callable[..., None], Tuple]):
    self.on = True
    self.callbacks = callbacks
    
  def stop(self):
    self.reset()
    self.on = False
    
  def reset(self):
    self.xs = []
    self.ys = []
    xlim = self.ax.get_xlim()
    ylim = self.ax.get_ylim()
    title = self.ax.get_title()
    self.ax.cla()
    self.ax.set_xlim(xlim)
    self.ax.set_ylim(ylim)
    self.ax.set_title(title)

  def __call__(self, event):
    if len(self.xs) == 2 or len(self.ys) == 2:
      self.reset()
    if type(event) == MouseEvent:
      if self.mode == 'normal':
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
      elif self.mode == 'x':
        self.ys.append(event.ydata)
        self.ys.append(event.ydata)
        xlim = self.ax.get_xlim()
        self.xs = list(xlim)
      elif self.mode == 'y':
        self.xs.append(event.xdata)
        self.xs.append(event.xdata)
        ylim = self.ax.get_ylim()
        self.ys = list(ylim)
      if len(self.xs) == 2 and len(self.ys) == 2:
        self.line = self.ax.plot(self.xs, self.ys)
    elif type(event) == KeyEvent:
      if event.key == 'n':
        self.mode = 'normal'
      elif event.key =='x':
        self.mode ='x'
      elif event.key =='y':
        self.mode ='y'
      else:
        pass
      time.sleep(0.5)
    
    if event.inaxes != None:
      for callback in self.callbacks:
        if len(callback) == 2:
          function = callback[0]
          params = callback[1]
          function(self, *params)
        elif len(callback) == 1:
          function = callback[0]
          function(self)
        
    self.fig.canvas.draw()

  def __setattr__(self, __name: str, __value) -> None:
    self.__dict__[__name] = __value
    if __name == 'on':
      if self.on:
        self.cid_1 = self.fig.canvas.mpl_connect('button_press_event', self)
        self.cid_2 = self.fig.canvas.mpl_connect('key_press_event', self)
      else:
        if self.cid_1:
          self.fig.canvas.mpl_disconnect(self.cid_1)
          self.cid_1 = None
        if self.cid_2:
          self.fig.canvas.mpl_disconnect(self.cid_2)
          self.cid_2 = None
        
        
# class LineDrawerSync(LineDrawer):
#   def __init__(self, fig, ax, on=True, callback=None):
#     super().__init__(fig, ax, on, callback)
#     self.ax = None
#     self.axs = [a for a in ax]
#     self.lines = []
    
#   def __call__(self, event):
#     xlim = [ax.get_xlim() for ax in self.axs if event.inaxes == ax]
#     ylim = [ax.get_ylim() for ax in self.axs if event.inaxes == ax]
#     if len(self.xs) == 2:
#       self.xs = []
#       self.ys = []
#       for (i, ax) in enumerate(self.axs):
#         ax.cla()
#         ax.set_xlim(xlim[0])
#         ax.set_ylim(ylim[0])
#     self.xs.append(event.xdata)
#     self.ys.append(event.ydata)
#     for ax in self.axs:
#       self.lines.append(ax.plot(self.xs, self.ys))
#     self.callback(self)
#     if len(self.xs) == 2:
#       for (i, ax) in enumerate(self.axs):
#           ax.set_xlim(xlim[0])
#           ax.set_ylim(ylim[0])
#     self.fig.canvas.draw()
  
#   def __setattr__(self, __name: str, __value) -> None:
#     return super().__setattr__(__name, __value)


class LineDrawerAsync(LineDrawer):
  def __init__(self, fig, ax) -> None:
    super().__init__(fig, ax)
    
  def __call__(self, event):
    if event.inaxes is self.ax:
      return super().__call__(event)
    pass
  
  def __setattr__(self, __name: str, __value) -> None:
    return super().__setattr__(__name, __value)


if __name__ == '__main__':
  fig, ax = plt.subplots(1, 2)
  for a in ax:
    a.set_xlim((0., 4.))
    a.set_ylim((0., 4.))
  line_drawer_1 = LineDrawer(fig, ax[0])
  line_drawer_2 = LineDrawer(fig, ax[1])
  def on_key_press(event):
    sys.stdout.flush()
    print(event.key)
    if event.key == 'd':
      line_drawer_1.draw()
      line_drawer_2.draw()
    elif event.key == 'c':
      line_drawer_1.stop()
      line_drawer_2.stop()
    else:
      pass
    time.sleep(0.05)
  def on_axes_enter(event):
      sys.stdout.flush()
      print(event)
      print('x: ', event.x, ', y: ', event.y)
      print('xdata: ', event.xdata, ', ydata: ', event.ydata)
  fig.canvas.mpl_connect('key_press_event', on_key_press)
  # fig.canvas.mpl_connect('axes_enter_event', on_axes_enter)

  plt.show()
