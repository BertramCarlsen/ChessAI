from __future__ import print_function
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

class Valuator(object): # Old Valuator
  def __init__(self):
    vals = torch.load("nets/value.pth", map_location=lambda storage, loc: storage)
    self.model = Net()
    self.model.load_state_dict(vals)

  def __call__(self, s):
    brd = s.serialize()[None]
    output = self.model(torch.tensor(brd).float())
    return float(output.data[0][0])
  
MAXVAL = 10000
class ClassicValuator(object):
  values = {chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0}

  def __init__(self):
    self.reset()
    self.memo = {}

  def reset(self):
    self.count = 0

  # good ideas:
  # https://en.wikipedia.org/wiki/Evaluation_function#In_chess
  def __call__(self, s):
    self.count += 1
    key = s.key()
    if key not in self.memo:
      self.memo[key] = self.value(s)
    return self.memo[key]

  def value(self, s):
    b = s.board
    # game over values
    if b.is_game_over():
      if b.result() == "1-0":
        return MAXVAL
      elif b.result() == "0-1":
        return -MAXVAL
      else:
        return 0

    val = 0.0
    # piece values
    pm = s.board.piece_map()
    for x in pm:
      tval = self.values[pm[x].piece_type]
      if pm[x].color == chess.WHITE:
        val += tval
      else:
        val -= tval

    # add a number of legal moves term
    bak = b.turn
    b.turn = chess.WHITE
    val += 0.1 * b.legal_moves.count()
    b.turn = chess.BLACK
    val -= 0.1 * b.legal_moves.count()
    b.turn = bak

    return val

def computer_minimax(s, v, depth, a, b, big=False):
  if depth >= 5 or s.board.is_game_over():
    return v(s)
  # white is maximizing player
  turn = s.board.turn
  if turn == chess.WHITE:
    ret = -MAXVAL
  else:
    ret = MAXVAL
  if big:
    bret = []

  # can prune here with beam search
  isort = []
  for e in s.board.legal_moves:
    s.board.push(e)
    isort.append((v(s), e))
    s.board.pop()
  move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)

  # beam search beyond depth 3
  if depth >= 3:
    move = move[:10]

  for e in [x[1] for x in move]:
    s.board.push(e)
    tval = computer_minimax(s, v, depth+1, a, b)
    s.board.pop()
    if big:
      bret.append((tval, e))
    if turn == chess.WHITE:
      ret = max(ret, tval)
      a = max(a, ret)
      if a >= b:
        break  # b cut-off
    else:
      ret = min(ret, tval)
      b = min(b, ret)
      if a >= b:
        break  # a cut-off
  if big:
    return ret, bret
  else:
    return ret

def explore_leaves(s, v):
  ret = []
  start = time.time()
  v.reset()
  bval = v(s)
  cval, ret = computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
  eta = time.time() - start
  print("%.2f -> %.2f: explored %d nodes in %.3f seconds %d/sec" % (bval, cval, v.count, eta, int(v.count/eta)))
  return ret

# Chess board + Engine
# v = Valuator()
s = State()
v = ClassicValuator()

def to_svg(s):
  return base64.b64encode(chess.svg.board(board=s.board).encode("utf-8")).decode("utf-8")

app = Flask(__name__)

@app.route("/")
def hello():
  ret = open("index.html").read()
  return ret.replace('start', s.board.fen())

  """ Commented for better HTML implementation
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
  """


def computer_move(s, v):
  # Computer move
  move = sorted(explore_leaves(s, v), key=lambda x: x[0], reverse=s.board.turn)
  if len(move) == 0:
    return
  print("Top 3 moves:")
  for i,m in enumerate(move[0:3]):
    print("  ", m)
  print(s.board.turn, "moving", move[0][1])
  s.board.push(move[0][1])

@app.route("/move")
def move():
  if not s.board.is_game_over():
    move = request.args.get("move", default="")
    if move is not None and move != "":
      print("Current FEN:", s.board.fen())
      print("Legal moves:", [s.board.san(m) for m in s.board.legal_moves])

      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
        print("Wrong move")
      response = app.response_class(
        response=s.board.fen(),
        status=200
      )
      return response
    else:
      print("Game is over")
      response = app.response_class(
        response="game over",
        status=200
      )
      return response
  print("hello ran")
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

@app.route("/move_coordinates")
def move_coordinates():
  if not s.board.is_game_over():
    source = int(request.args.get('from', default=''))
    target = int(request.args.get('to', default=''))
    promotion = True if request.args.get('promotion', default='') == 'true' else False

    move = s.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))

    # Debugging steps
    print("Current FEN:", s.board.fen())  
    print("Legal moves:", [m.uci() for m in s.board.legal_moves])
    print(f"Attempting move: {move.uci()}")

    if move is not None and move != "":
      print("human moves", move)
      try:
        s.board.push_san(move)
        computer_move(s, v)
      except Exception:
        traceback.print_exc()
    response = app.response_class(
      response=s.board.fen(),
      status=200
    )
    return response

  print("GAME IS OVER")
  response = app.response_class(
    response="game over",
    status=200
  )
  return response

@app.route("/newgame")
def newgame():
  s.board.reset()
  response = app.response_class(
    response=s.board.fen(),
    status=200
  )
  return response


if __name__ == "__main__":
  if os.getenv("SELFPLAY") is not None:
    s = State()
    while not s.board.is_game_over():
      computer_move(s, v)
      print(s.board)
    print(s.board.result())
  else:
    app.run(debug=True)