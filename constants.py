

# TeamConst
""" Defining the constants for agents and teams """
RED = 10
BLUE = 50
GRAY = 90
NUM_BLUE = 4
NUM_RED = 4
NUM_UAV = 2
NUM_GRAY = 10
UAV_STEP = 3
UGV_STEP = 1
UAV_RANGE = 4
UGV_RANGE = 2
UAV_A_RANGE = 0
UGV_A_RANGE = 2


# MapConst
""" Defining the constants for map and environment """
WORLD_H = 100
WORLD_W = 100
RED_ZONE = 15
RED_AGENT = 20
RED_FLAG = 10
BLUE_ZONE = 55
BLUE_AGENT = 60
BLUE_FLAG = 50
GRAY_AGENT = 95
#OBSTACLE = 100
AERIAL_DENIAL = 90

UNKNOWN = -1
TEAM1_BACKGROUND = 0
TEAM2_BACKGROUND = 1
TEAM1_UGV = 2
TEAM1_UAV = 3
TEAM2_UGV = 4
TEAM2_UAV = 5
TEAM1_FLAG = 6
TEAM2_FLAG = 7
OBSTACLE = 8
DEAD = 9

COLOR_DICT = {UNKNOWN : (200, 200, 200),
              TEAM1_BACKGROUND : (0, 0, 120),
              TEAM2_BACKGROUND : (120, 0, 0),
              TEAM1_UGV : (0, 0, 255),
              TEAM1_UAV : (0, 0, 255),
              TEAM2_UGV : (255, 0, 0),
              TEAM2_UAV :  (255, 0, 0),
              TEAM1_FLAG : (0, 255, 255),
              TEAM2_FLAG : (255, 255, 0),
              OBSTACLE : (120, 120, 120),
              DEAD : (0, 0, 0)}

# _ means assignment is independent of a
# in the current implementation enemy vehicle dont cross the border
operator_dict= {
    (UNKNOWN, UNKNOWN): UNKNOWN,

    ('xx', TEAM1_FLAG): TEAM1_FLAG,
    ('xx', TEAM2_FLAG): TEAM2_FLAG,
    ('xx', OBSTACLE): OBSTACLE,
    ('xx', TEAM2_FLAG): TEAM2_FLAG,
    ('xx', TEAM1_UGV): UNKNOWN,
    ('xx', TEAM1_UAV): UNKNOWN,
    ('xx', TEAM1_BACKGROUND): TEAM1_BACKGROUND,
    ('xx', TEAM2_BACKGROUND): TEAM2_BACKGROUND,
    ('xx', TEAM2_UGV): TEAM2_BACKGROUND,
    ('xx', TEAM2_UAV): TEAM2_BACKGROUND,
    (OBSTACLE, 'xx'): OBSTACLE,
    (TEAM1_BACKGROUND, 'xx'): TEAM1_BACKGROUND,
    (TEAM2_BACKGROUND, 'xx'): TEAM2_BACKGROUND,

}
