import math
import time
import colored
import threading
import numpy as np

import stt_assist

class piece:
    def __init__(self, piece=0, color=2, style_filled=True):
        # style_filled True: filled pieces
        # style_filled False: outlined pieces

        # color 0: white
        # color 1: black
        # color 2: empty cell's neutral color
        self.empty_cell = ' '
        self.color = color
        self.white_fg = colored.fore_rgb(150, 0, 150)
        self.black_fg = colored.fore_rgb(0, 100, 200)
        self.black_cell_bg = colored.back_rgb(200, 180, 0)
        self.white_cell_bg = colored.back_rgb(0, 0, 105)

        self.pieces_unicode = np.array([
            '♔',
            '♕',
            '♖',
            '♗',
			'♘',
			'♙',
			'♚',
			'♛',
			'♜',
			'♝',
			'♞',
			'♟'
        ])

        if piece == -1:
            self.piece_unicode = self.empty_cell
        else:
            # self.piece_unicode = self.pieces_unicode[5-piece+6*color]

            if piece == 0:
                # Conditional unicode symbol assignment to pawn using color
                # self.piece_unicode = self.pieces_unicode[5-piece+6*color]
                self.piece_unicode = self.pieces_unicode[5-piece+6*0]
            else:
                self.piece_unicode = self.pieces_unicode[5-piece+6*style_filled]

        self.pieces = {
            -1: [None, None], # empty
            0: ["pawn", 1],
            1: ["knight", 3],
            2: ["bishop", 3],
            3: ["rook", 5],
            4: ["queen", 9],
            5: ["king", float('inf')],
        }
        self.piece = self.pieces.get(piece)
        if self.piece is None:
            self.piece = self.pieces[0]

    def __str__(self, short=True, disp=True, show_color=False):
        if self.piece_unicode == self.empty_cell:
            return self.piece_unicode
        if disp:
            if show_color:
                return f"{[colored.Fore.WHITE, colored.Fore.BLACK][self.color]}{(self.piece_unicode)}{colored.Style.reset}"
            else:
                return f"{(self.piece_unicode)}"
        if short:
            return f"{['w', 'b'][self.color]}{self.piece[0][0]}[{self.piece[1]}]"
        return f"{self.piece[0]}('{['white', 'black'][self.color]}', {self.piece[1]})[{self.piece_unicode}]"

class chess_board(piece):
    def __init__(self, top_color=0):
        super().__init__(-1)
        self.top_color = top_color
        self.override_colors = {
            # 'valid': colored.back_rgb(0, 90, 200)+colored.fore_rgb(0, 0, 0),
            'valid': colored.back_rgb(0, 180, 0)+colored.fore_rgb(0, 0, 0),
            'valid_alt': colored.back_rgb(0, 100, 0)+colored.fore_rgb(255, 255, 20),
            'last_moved_src': colored.back_rgb(160, 120, 0),
            'last_moved_dest': colored.back_rgb(200, 170, 0),
            'valid_capture': colored.back_rgb(170, 0, 70)+colored.fore_rgb(0, 0, 0),
            'last_captured': colored.back_rgb(120, 80, 0),
            'check': colored.back_rgb(200, 100, 0),
            'mate': colored.back_rgb(150, 0, 0),
            'sel': colored.back_rgb(0, 255, 0),
        }

        # Top color is white (0) by default
        self.reset_board(top_color)
        # self.custom_layout(top_color)

        # Helpers
        self.db = self.display_board
        self.sb = self.stringified_board

    # Helper Functions
    def disp(self, x, y):
        return self.board[x, y].__str__()

    def look(self, x, y):
        return self.board[x, y].__str__(disp=False)

    def custom_layout(self, layout, top_color=0):
        self.board = np.array([
            [
                piece(4, top_color),
                piece(4, top_color),
                piece(4, top_color),
                piece(4, top_color),
                piece(4, top_color),
                piece(4, top_color),
                piece(4, top_color),
                piece(4, top_color),
            ],
            [piece(0, top_color)] * 8,

            # [piece(-1)] * 8,

            [piece(-1)] * 8,
            [piece(-1)] * 8,
            [piece(-1)] * 8,
            [piece(-1)] * 8,

            # [piece(-1)] * 8,

            [piece(0, not top_color)] * 8,
            [
                piece(0, not top_color),
                piece(0, not top_color),
                piece(0, not top_color),
                piece(0, not top_color),
                piece(0, not top_color),
                piece(0, not top_color),
                piece(0, not top_color),
                piece(0, not top_color),
            ],
        ])


    def reset_board(self, top_color=0):
        self.board = np.array([
            [
                piece(3, top_color),
                piece(2, top_color),
                piece(1, top_color),
                piece(4, top_color),
                piece(5, top_color),
                piece(1, top_color),
                piece(2, top_color),
                piece(3, top_color)
            ],
            # [piece(0, top_color)] * 8,

            [piece(-1)] * 8,

            [piece(-1)] * 8,
            [piece(-1)] * 8,
            [piece(-1)] * 8,
            [piece(-1)] * 8,

            [piece(-1)] * 8,

            # [piece(0, not top_color)] * 8,
            [
                piece(3, not top_color),
                piece(2, not top_color),
                piece(1, not top_color),
                piece(4, not top_color),
                piece(5, not top_color),
                piece(1, not top_color),
                piece(2, not top_color),
                piece(3, not top_color)
            ],
        ])

    def display_board(self,
                      current_idx=None,
                      override_color=None,
                      override_cells=None):
        # for i in self.stringified_board():
        #     print(' '.join(i))

        # csr = colored.Style.reset
        print(' ', ' '.join(list('abcdefgh')))
        # print(f"{override_color.__repr__()}")
        # print(f"{override_color}test{csr}")

        bgs = [self.white_cell_bg, self.black_cell_bg]
        fgs = [self.white_fg, self.black_fg, colored.Fore.WHITE]
        valid_empty_cell_symbol = '\u25A0\u25A0'
        valid_empty_cell_symbol = 'xx'

        valid_alt = self.override_colors['valid_alt']

        for idx, i in enumerate(self.stringified_board()):
            print(9-(idx+1), end=' ')
            for jdx, j in enumerate(i):
                fg = self.board[idx, jdx].color
                if override_cells:
                    if (idx, jdx) in override_cells:
                        if self.board[idx, jdx].piece_unicode == self.empty_cell:

                            if not (jdx % 2):
                                if not (idx % 2):
                                    # Alt Valid Move
                                    print(f"{valid_alt}{valid_empty_cell_symbol}{valid_alt}", end='')
                                else:
                                    print(f"{override_color}{valid_empty_cell_symbol}{override_color}", end='')
                            else:
                                if not (idx % 2):
                                    # Alt Valid Move
                                    print(f"{override_color}{valid_empty_cell_symbol}{override_color}", end='')
                                else:
                                    print(f"{valid_alt}{valid_empty_cell_symbol}{valid_alt}", end='')

                                # Valid Move
                                # print(f"{override_color}{valid_empty_cell_symbol}{override_color}", end='')
                            # print(f"{valid_empty_cell_color}{override_color}{valid_empty_cell_symbol}{valid_empty_cell_color}{override_color}", end='')
                        else:
                            # print(f"{self.override_colors['valid_capture']}■■", end='')
                            print(f"{self.override_colors['valid_capture']}{self.board[idx, jdx]} ", end='')
                        continue
                    elif current_idx and (idx, jdx) == current_idx:
                        print(f"{fgs[fg]}{self.override_colors['sel']}{j}{fgs[fg]}{self.override_colors['sel']} ", end='')
                        continue
                    # continue
                print(f"{fgs[fg]}{bgs[(idx+jdx)%2]}{j} {fgs[fg]}{bgs[(idx+jdx)%2]}", end='')
            print(colored.Style.reset)

        return i, j

    def __str__(self):
        return str(self.stringified_board())

    def stringified_board(self):
        return np.vectorize(str)(self.board)


class game_play:
    def __init__(self,
                 chess_board,
                 start_turn = 0,
                 game_duration = 3*60*1000,
                 is_timed = True,
                 game_type = None,
                 ask_confirms = [True, True]):
        # start_turn 0: white
        # start_turn 1: black
        self.cb = chess_board
        self.turn = start_turn
        self.game_type = game_type
        self.ask_confirms = ask_confirms

        self.white_pieces_captured = []
        self.black_pieces_captured = []

        if is_timed:
            print('sorry, timed chess is still in progress')
            self.game_duration = game_duration
            self.wclock = self.game_duration
            self.bclock = self.game_duration

    # def start_clock(self, turn):
    #     if turn == 0:
    #         self.wclock

    def get_piece_idx(self, index):
        index = index.lower()
        if not (len(index)==2 and index[0].isalpha() and index[1].isnumeric()):
            raise ValueError(f"Invalid Index {index}")

        try:
            c, r = index
            r = int(r)
        except Exception:
            raise ValueError(f"Invalid Index {index}")

        if not (c in list('abcdefgh') and r in range(1, 9)):
            raise ValueError(f"Invalid Index {index}")

        c = ord(c) - ord('a')
        r = 8 - r
        
        return r, c


    def is_valid_piece_selected(self, idx):
        piece = self.cb.board[idx]
        # print('piece', piece.color, piece.piece_unicode)
        # print('turn', self.turn)
        if piece.piece_unicode == self.cb.empty_cell:
            return False
        if piece.color == self.turn:
            # TODO: Check for dead end piece
            # TODO: or protecting king from checking
            return True
        else:
            return False

    def get_valid_moves(self, idx):
        # TODO: check for castling
        # TODO: check for en-passant

        valid_moves = []
        valid_capture_moves = []
        x, y = idx
        piece = self.cb.board[idx]

        if piece.piece[0] == 'pawn':
            if self.turn != self.cb.top_color:
                # current player at bottom (normal pov)
                if self.cb.board[x-1, y].piece_unicode == self.cb.empty_cell:
                    valid_moves.append((x-1, y))
                if self.cb.board[x-2, y].piece_unicode == self.cb.empty_cell and x == 6:
                    valid_moves.append((x-2, y))
                
                try:
                    if self.cb.board[x-1, y-1].piece_unicode != self.cb.empty_cell and \
                       self.cb.board[x-1, y-1].color != self.cb.board[x, y].color:
                        valid_moves.append((x-1, y-1))
                except IndexError:
                    pass

                try:
                    if self.cb.board[x-1, y+1].piece_unicode != self.cb.empty_cell and \
                       self.cb.board[x-1, y+1].color != self.cb.board[x, y].color:
                        valid_moves.append((x-1, y+1))
                except IndexError:
                    pass
            else:
                pass

        if piece.piece[0] == 'bishop':
            if self.turn != self.cb.top_color:
                i, j = x, y
                while i>=0 and j>=0:
                    i, j = i - 1, j - 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                        break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                while i<=6 and j>=0:
                    i, j = i + 1, j - 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                        break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                while i>=0 and j<=6:
                    i, j = i - 1, j + 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                        break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                while i<=6 and j<=6:
                    i, j = i + 1, j + 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                        break
                    else:
                        valid_moves.append((i, j))
            else:
                pass

        if piece.piece[0] == 'knight':
            if self.turn != self.cb.top_color:
                combs = [
                    (x-2, y-1), (x-2, y+1),
                    (x-1, y-2), (x-1, y+2),
                    (x+1, y-2), (x+1, y+2),
                    (x+2, y-1), (x+2, y+1),
                ]
                combs = [x for x in combs if all(i<8 for i in x)]
                valid_moves = [
                    i for i in combs if
                    self.cb.board[i].piece_unicode == self.cb.empty_cell or
                    self.cb.board[i].color != self.cb.board[x, y].color
                ]
            else:
                pass

        if piece.piece[0] == 'rook':
            if self.turn != self.cb.top_color:
                i, j = x, y
                # capturable = False
                while i >= 0:
                    i -= 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            # if capturable: break
                            # capturable = True
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                i = x
                # capturable = False
                while i <= 6:
                    i += 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            # if capturable: break
                            # capturable = True
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                # capturable = False
                while j >= 0:
                    j -= 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            # if capturable: break
                            # capturable = True
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                j = y
                # capturable = False
                while j <= 6:
                    j += 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            # if capturable: break
                            # capturable = True
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))
            else:
                pass

        if piece.piece[0] == 'queen':
            if self.turn != self.cb.top_color:

                # Diagonal Moves
                i, j = x, y
                while i>=0 and j>=0:
                    i, j = i - 1, j - 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                while i<=6 and j>=0:
                    i, j = i + 1, j - 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                while i>=0 and j<=6:
                    i, j = i - 1, j + 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                while i<=6 and j<=6:
                    i, j = i + 1, j + 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                # Straight Moves
                i, j = x, y
                while i >= 0:
                    i -= 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                i = x
                while i <= 6:
                    i += 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                i, j = x, y
                while j >= 0:
                    j -= 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))

                j = y
                while j <= 6:
                    j += 1
                    if self.cb.board[i, j].piece_unicode != self.cb.empty_cell:
                        if self.cb.board[i, j].color != self.cb.board[x, y].color:
                            valid_capture_moves.append((i, j))
                            valid_moves.append((i, j))
                            break
                    else:
                        valid_moves.append((i, j))
            else:
                pass

        if piece.piece[0] == 'king':
            if self.turn != self.cb.top_color:
                combs = [
                    (x-1, y-1), (x-1, y), (x-1, y+1),
                    (x  , y-1),           (x  , y+1),
                    (x+1, y-1), (x+1, y), (x+1, y+1),
                ]
                combs = [x for x in combs if all(i<8 for i in x)]
                valid_moves = [i for i in combs if self.cb.board[i].piece_unicode == self.cb.empty_cell]
            else:
                pass

        return valid_moves

    def disp_mark_valid_moves(self, current_idx, valid_moves):
        # print(valid_moves)
        cb.display_board(
            current_idx=current_idx,
            override_color=cb.override_colors['valid'],
            override_cells=valid_moves
        )

    def mark_move(self, idx, dest_idx):
        # TODO: mark/play the move (validity has already been checked)
        # TODO: capture any piece if present
        # TODO: change turn -> not turn
        # TODO: check for upgrading pawn after reaching end

        # if not self.ask_confirms[turn]:
        #     # play the move
        #     pass

        if self.cb.board[idx].piece_unicode == self.cb.empty_cell:
            return

        self.cb.board[idx], self.cb.board[dest_idx] = self.cb.board[dest_idx], self.cb.board[idx]

        return self.cb.board

    def select_piece(self, index):
        piece_idx = self.get_piece_idx(index)
        # print(piece_idx, self.is_valid_piece_selected(piece_idx))
        if self.is_valid_piece_selected(piece_idx):
            valid_moves = self.get_valid_moves(piece_idx)
            self.disp_mark_valid_moves(piece_idx, valid_moves)

            return piece_idx, valid_moves
        return piece_idx

class main_loop:
    # Start recording until a pause

    def stt(self):
        fpath = stt_assist.record_until_pause()
        # audio_file = "./tmp/usr_aud.mp3"
        transcription = stt_assist.transcribe_audio(fpath)
        notation = stt_assist.rule_based(transcription)
        return notation

def main():
    global cb, a, gp, p, m, mm
    cb = chess_board()
    # a = piece(1)

    # Game with black's start
    gp = game_play(cb, start_turn=1, is_timed=False)
    # p, m = gp.select_piece('c2')

    # gp.mark_move(p, m[1])

    # BOARD TESTING
    piece_id = 'e5'
    # piece_id = 'f6'
    for i in range(1, 5):
        cb = chess_board()
        gp = game_play(cb, start_turn=1, is_timed=False)
        mm = gp.mark_move((7, i), gp.get_piece_idx(piece_id))
        p, m = gp.select_piece(piece_id)
        print()

    cb = chess_board()
    gp = game_play(cb, start_turn=1, is_timed=False)
    mm = gp.mark_move((6, i), gp.get_piece_idx(piece_id))
    p, m = gp.select_piece(piece_id)
    print()

    piece_id = 'f6'
    cb = chess_board()
    gp = game_play(cb, start_turn=1, is_timed=False)
    mm = gp.mark_move((6, i), gp.get_piece_idx(piece_id))
    p, m = gp.select_piece(piece_id)
    print()

    # f = colored.fore_rgb(150, 0, 150)
    # b = colored.back_rgb(200, 180, 0)
    # r = colored.Style.reset
    # print(f, b, 'test', r)
    

    # cb.db()

if __name__ == "__main__":
    main()


