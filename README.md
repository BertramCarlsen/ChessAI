# Chess AI

A deep learning-based chess engine that combines neural network position evaluation with minimax search algorithm. Play against an AI that has learned chess position evaluation from real games.

## Features

- **Neural Network Evaluation**: Deep CNN trained on chess game data for position evaluation
- **Minimax Search**: Alpha-beta pruned minimax algorithm with beam search optimization
- **Web Interface**: Interactive chess board with drag-and-drop moves
- **Self-Play Mode**: Watch the AI play against itself
- **Training Visualization**: Real-time loss plots during neural network training

## Architecture

### Neural Network (`neural.py`)
- **Input**: 5-channel 8x8 representation of chess board
  - 4 channels for piece encoding (binary representation)
  - 1 channel for turn information
- **Architecture**: Convolutional Neural Network with progressive downsampling
  - Conv layers: 5→16→16→32→32→32→64→64→64→128→128→128
  - Output: Single value (-1 to 1) representing position evaluation
- **Training**: Uses SmoothL1Loss with Adam optimizer

### Search Algorithm (`playchess.py`)
- **Minimax with Alpha-Beta Pruning**: 5-ply search depth
- **Beam Search**: Top 10 moves considered at depth ≥3 for efficiency
- **Move Ordering**: Positions pre-evaluated for better pruning
- **Fallback Evaluator**: Classical piece-value evaluation as backup

## Setup

### Requirements
```bash
pip install torch numpy matplotlib chess flask mpld3
```

### Data Preparation
1. Place PGN files in `data/` directory
2. Run dataset generation:
```bash
python get_dataset.py
```
This creates `processed/dataset_10M.npz` with serialized positions and game outcomes.

### Training
```bash
python neural.py
```
- Trains for 100 epochs with batch size 256
- Saves model weights to `nets/value.pth`
- Generates training loss visualization in `static/loss_plot.html`

### Running the Web Interface
```bash
python playchess.py
```
Navigate to `http://localhost:5000` to play against the AI.

## File Structure

```
├── neural.py          # Neural network definition and training
├── playchess.py       # Flask web app and game engine
├── state.py           # Chess board state serialization
├── get_dataset.py     # PGN data processing
├── index.html         # Web interface
├── data/              # PGN files directory
├── processed/         # Processed datasets
├── nets/              # Trained model weights
└── static/            # CSS, JS, and visualization files
```

## How It Works

### Board Representation
Chess positions are encoded as 5×8×8 tensors:
- **Channels 0-3**: Binary encoding of piece types and colors
- **Channel 4**: Current player to move
- **Special encoding**: Castling rights and en passant squares

### Position Evaluation
1. **Neural Network**: Outputs value between -1 (bad for white) and +1 (good for white)
2. **Classical Backup**: Traditional piece values + mobility evaluation

### Move Selection
1. Generate all legal moves
2. Evaluate each position using neural network
3. Apply minimax search with alpha-beta pruning
4. Select move with best evaluation

## API Endpoints

- `GET /` - Main game interface
- `GET /move?move=<san>` - Make a move in algebraic notation
- `GET /move_coordinates?from=<sq>&to=<sq>` - Make a move by coordinates
- `GET /newgame` - Start a new game
- `GET /selfplay` - Watch AI vs AI

## Training Data Format

The neural network expects data in the format:
- **X**: Array of shape (N, 5, 8, 8) - board positions
- **Y**: Array of shape (N,) - game outcomes (-1, 0, 1)

## Performance

- **Search Speed**: ~1000-10000 positions/second (depending on hardware)
- **Search Depth**: 5 plies with beam search pruning
- **Training Time**: ~100 epochs on 10K-10M positions

## Customization

### Adjusting AI Strength
- Modify `depth` parameter in `computer_minimax()` function
- Adjust beam search width (currently top 10 moves)
- Change evaluation function weights

### Neural Network Architecture
- Modify the `Net` class in `neural.py`
- Adjust hyperparameters (learning rate, batch size, epochs)

## Self-Play Mode
Visit `/selfplay` to watch the AI play complete games against itself.

## Acknowledgments

- Uses the `python-chess` library for chess logic
- Web interface built with ChessBoard.js
- Neural network implemented with PyTorch