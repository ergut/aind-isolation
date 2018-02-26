"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np
from numpy import linalg as LA


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score_10(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Using the difference in num of available moves as a score metric
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    def score(moves):
        #TODO: check order, height or width first?
        max_norm = (game.height + game.width)/2.0
        if len(moves) == 0:
            return -np.inf
        # Negative infinity norm (min value)
        norm = LA.norm(np.array(moves) - (game.height/2.0, game.width/2.0), -np.inf)
        # the smaller the norm, the greater the score
        return max_norm - norm

    return float(score(own_moves) - score(opp_moves))


def get_2step_moves(game, player):
    """ Returns one step and two step legal moves for a given player

    :param game: Board class
    :param player: Player (own or opponent)
    :return: one_step_moves, two_step_moves
    """
    one_step_moves = game.get_legal_moves(player)
    two_step_moves = []
    for move in one_step_moves:
        game._active_player = player
        two_step_moves.append(game.forecast_move(move).get_legal_moves(player))

    # Remove repeating ones (flatten using sum)
    two_step_moves = list(set(sum(two_step_moves,[])))

    return one_step_moves, two_step_moves

def get_score(game, player, k1, k2, k3, k4):
    opp = game.get_opponent(player)

    own_one_step_moves, own_two_step_moves = get_2step_moves(game.copy(), player)
    opp_one_step_moves, opp_two_step_moves = get_2step_moves(game.copy(), opp)

    common_one_step_moves = [x for x in own_one_step_moves if x in opp_one_step_moves]
    common_two_step_moves = [x for x in own_two_step_moves if x in opp_two_step_moves]

    score = k1 * (len(own_one_step_moves) - len(opp_one_step_moves)) + \
            k2 * (len(own_two_step_moves) - len(opp_two_step_moves)) + \
            k3 * len(common_one_step_moves) + \
            k4 * len(common_two_step_moves)
    return score

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Using the difference in num of available moves as a score metric
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    return float(get_score(game, player, k1=1, k2=0.6, k3=0.8, k4=0.5))

def custom_score_11(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Using the difference in num of available moves as a score metric
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    return float(get_score(game, player, k1=1, k2=0.6, k3=0.8, k4=-0.2))

def custom_score_8(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # Using the difference in num of available moves as a score metric
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    if (sum(game._board_state[:-3]) / len(game._board_state[:-3])) < 0.6:
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return float(own_moves - opp_moves)

    else:
        MAX_DEPTH = 4
        def calculate_depth(game, player, depth, max_depth):

            moves = game.get_legal_moves(player)
            if len(moves) == 0:
                return depth ##FIXME: 0???

            if depth >= max_depth:
                # There are moves available but we reached our limit
                return depth+0.5

            d_max = depth
            for move in moves:
                game._active_player = player
                d = calculate_depth(game.forecast_move(move), player, depth+1, max_depth)
                d_max = max(d_max, d)
                if d_max >= max_depth:
                    # Already found max depth no need to continue
                    break
                game._board_state[-3] ^= 1
                #game._active_player, game._inactive_player = game._inactive_player, game._active_player

            return d_max

        opp = game.get_opponent(player)
        d_opp = calculate_depth(game.copy(), opp, 0, MAX_DEPTH)
        d_own = calculate_depth(game.copy(), player, 0, min(MAX_DEPTH, d_opp))

    return float(d_own - d_opp)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # Using the difference in num of available moves as a score metric
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    common_moves = [x for x in own_moves if x in opp_moves]
    # Opponent can block at least one of the common moves
    return float(len(own_moves) - len(opp_moves) - (1 if len(common_moves)>0 else 0))

def custom_score_6(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    def get_score(moves):
        n_level2_moves = 0
        for a in moves:
            legal_moves = game._Board__get_moves(a)
            n_level2_moves += len(legal_moves)
        if n_level2_moves > 0:
            score = 10 + n_level2_moves
        else:
            # There is no level 2 moves, having 1 or 3 alternatives is the same
            score = 1
        return score

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    own_score = get_score(own_moves)
    #opp_score = get_score(opp_moves)
    opp_score = len(opp_moves)

    return float(own_score - opp_score)



def custom_score_5(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    def get_score(moves):
        n_level2_moves = 0
        best_move = []
        for a in moves:
            legal_moves = game._Board__get_moves(a)
            n_legal_moves = len(legal_moves)
            if n_legal_moves > n_level2_moves:
                n_level2_moves = n_legal_moves
                best_move = a
        if n_level2_moves > 0:
            score = 10 + n_level2_moves
        else:
            # There is no level 2 moves, having 1 or 3 alternatives is the same
            score = 1
        return score, best_move

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    common_moves = [x for x in own_moves if x in opp_moves]

    own_score, own_best_move = get_score(own_moves)
    if own_best_move in common_moves:
        # Best move will be taken away by the opponent. Remove it get the second best score
        own_score, _ = get_score([x for x in own_moves if x != own_best_move])

    opp_score = len(opp_moves)

    return float(own_score - opp_score)
def custom_score_4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")


    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    common_moves = [x for x in own_moves if x in opp_moves]

    return float(len(own_moves) - len(opp_moves) - 0.5* len(common_moves))


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    def calculate_score(moves):
        if len(moves) == 0:
            return 0
        centered_locs = np.array(moves) - (game.height/2.0, game.width/2.0)
        #print(centered_locs)
        return np.sum(1/LA.norm(centered_locs,ord=2,axis=1))

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    return float(calculate_score(own_moves)-calculate_score(opp_moves))


def custom_score_7(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    def calculate_score(moves):
        if len(moves) == 0:
            return 0
        centered_locs = np.array(moves) - (game.height/2.0, game.width/2.0)
        return min(np.sum(centered_locs,axis=1))

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    return float(calculate_score(own_moves)-calculate_score(opp_moves))



class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the bes   t move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            print('Timeout\n')
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()


        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        best_move = (-1, -1)
        val = -np.inf
        for move in legal_moves:
            v = self.min_value(game.forecast_move(move),depth-1)
            if v > val:
                val = v
                best_move = move

        return best_move

    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves or depth == 0:
            return self.score(game,self)
            # TODO: if no legal_moves, someone loses

        val = -np.inf
        for move in legal_moves:
            val = max(val, self.min_value(game.forecast_move(move), depth-1))
        return val

    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves or depth == 0:
            return self.score(game,self)

        val = np.inf
        for move in legal_moves:
            val = min(val, self.max_value(game.forecast_move(move), depth-1))

        return val

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            depth = self.search_depth
            while True: # TODO replace it with TRUE
                self.debug(depth,'depth={}'.format(depth))
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            self.debug(depth,'Timeout, depth = {}\n'.format(depth))
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move


    def debug(self, depth, str):
        return
        if depth < 0:
            raise ValueError('Negative depth value')
        level = self.search_depth - depth
        print('|   '*level + str)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        self.debug(depth, '******** ENTERING ALPHA-BETA-SEARCH *****************')

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            self.debug(depth, "no legal moves")
            return (-1, -1)

        best_move = (-1, -1)
        val = float('-inf')

        self.debug(depth, 'INITIAL stage moves = {}'.format(legal_moves))
        for move in legal_moves:
            self.debug(depth, 'a = {}'.format(move))
            v = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if v > val:
                val = v
                best_move = move
            alpha = max(alpha, val)
            if val >= beta:
                break

        return best_move

    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves or depth == 0:
            return self.score(game, self)
        val = float('-inf')
        for move in legal_moves:
            v = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            val = max(val, v)
            alpha = max(alpha, val)
            if val >= beta:
                break
        return val


    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves or depth == 0:
            return self.score(game, self)

        val = float('inf')
        for move in legal_moves:
            v = self.max_value(game.forecast_move(move), depth - 1, alpha, beta)
            val = min(val, v)
            beta = min(beta, val)
            if val <= alpha:
                break
        return val
