import tkinter as tk
from tkinter import *

from create_dataset import *
from test import *
from train import *

fontstyle = 'Ebrima'
bg_color = 'white'
bt_color = 'green'

root = tk.Tk()
root.geometry('800x400')
root.title('Sign language translator')
root.configure(bg=bg_color)
        
word = StringVar()

def return_word():
    get_word = word.get()
    create(get_word)


# 센터 정렬을 위한 빈 라벨
tk.Label(root, text="", width=15, height=10, bg=bg_color).grid(row=0, column=0)

# 타이틀
tk.Label(root, text="수화번역기", font=(fontstyle, 24), bg=bg_color).grid(row=0, column=2)

# 추가할 데이터셋 이름 입력 받기
tk.Label(root, text="학습할 말: ", font=(fontstyle, 18), bg=bg_color).grid(row=1, column=1)
tk.Entry(root, width=25, textvariable=word, font=(fontstyle, 18), bg=bg_color).grid(row=1, column=2)

# 데이터 수집 버튼
tk.Button(root, text="수집", font=(fontstyle, 15), width=10, command=return_word, bg=bt_color, fg=bg_color).grid(row=1, column=3)

# 학습 버튼
tk.Button(root, text="학습", font=(fontstyle, 15), width=20, bg=bt_color, command=train, fg=bg_color).grid(row=2, column=2, pady=5)

# 번역 버튼
tk.Button(root, text="수화 번역", font=(fontstyle, 15), width=20, command=test, bg=bt_color, fg=bg_color).grid(row=3, column=2, pady=5)
        
root.mainloop()

