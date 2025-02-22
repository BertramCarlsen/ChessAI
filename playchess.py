import torch
from state import State
from neural import Net
from flask import Flask, Response, request
import chess
import chess.svg
import time
import os
import traceback
import base64

class Valuator(object):
  def __init__(self):
    vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)

  def __call__(self, s):
    brd = s.serialize()[None]
    output = self.model(torch.tensor(brd).float())
    return float(output.data[0][0])

def explore_leaves(s, v):
  ret = []
  for e in s.edges():
    s.board.push(e)
    ret.append((v(s), e))
    s.board.pop()
  return ret

# Chess board + Engine
v = Valuator()
s = State()

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode("utf-8")).decode("utf-8")

app = Flask(__name__)

@app.route("/")
def hello():
  board_svg = to_svg(s)
  # Concatenating html lol
  ret = "<html><head>"
  ret += "<script src='https://ajax.googleapis.com/ajax/libs/jquery/3.7.1/jquery.min.js'></script>"
  ret += "<style>input { font-size: 30px; } button { font-size: 20px; }</style>"
  ret += "</head><body>"
  ret += "<img src='data:image/svg+xml;base64,%s' width='700' height='700' text-align='center'>" % board_svg
  ret += '<form action="/move"><input name="move" type="text"></input><input type="submit" value="Move"></form><br>'
  ret += "<a href='/selfplay'>Play vs itself</a><br/>"
  ret += "<script>$(function() { var input = document.getElementById('move'); console.log('selected'); input.focus(); input.select(); }); </script>"
  # ret += "<button onclick=\"$.post('/move', function() { location.reload(); });\">Make computer move</button>"
  return ret

def computer_move(s, v):
  # Computer move
  move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
  print("Top 3 moves:")
  for i,m in enumerate(move[0:3]):
    print(" ", m)
  s.board.push(move[0][1])

@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get("move", default="")
    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
        print("Wrong move")
    else:
      print("Game is over")
    return hello()

@app.route("/selfplay")
def selfplay():
  ret = "<html><head>"
  s = State()
  # Selfplay:
  while not s.board.is_game_over():
    computer_move(s, v)
    ret += "<img src='data:image/svg+xml;base64,%s' width='700' height='700' text-align='center'>" % to_svg(s)
  print(s.board.result())
  return ret

if __name__ == "__main__":
  app.run(debug=True)
