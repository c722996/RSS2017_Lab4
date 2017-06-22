#!/usr/bin/env python

import numpy as np 
import random 
import math
import utils as Utils
import Map
import tf.transformations
import tf

from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray
from nav_msgs.srv import GetMap

import rospy
import time
from Map import Map

class RRT():

    def __init__(self,environment): 
        
        # variables
        self.start = None
        self.goal = None
        self.map_info = None
        self.environment = environment

        # these topics are to receive data from the rviz
        self.start_sub  = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.startCB, queue_size=1)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goalCB, queue_size=1)

        # these topics are for visualization
        self.path_pub  = rospy.Publisher("/rrt/viz/path", Path, queue_size = 1)
        self.start_pub  = rospy.Publisher("/rrt/viz/start", PointStamped, queue_size = 1)
        self.goal_pub  = rospy.Publisher("/rrt/viz/goal", PointStamped, queue_size = 1)

    def get_omap(self):
        '''
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        '''
        # this way you could give it a different map server as a parameter
        map_service_name = rospy.get_param("~static_map", "static_map")
        print("getting map from service: ", map_service_name)
        rospy.wait_for_service(map_service_name)
        map_msg = rospy.ServiceProxy(map_service_name, GetMap)().map
        #rospy.loginfo('getting map_info')

        self.map_info = map_msg.info
        rospy.loginfo(self.map_info)

    def startCB(self,msg):
        #rospy.loginfo('recieved start msg')
        self.get_omap()
        self.start = np.array([(msg.pose.pose.position.x, msg.pose.pose.position.y,0)])
        Utils.world_to_map(self.start,self.map_info)
        self.start = np.array((self.start[0][0],1300-self.start[0][1],self.start[0][2])).astype(int)
        #rospy.loginfo(self.start)

    def goalCB(self,msg):
        #rospy.loginfo('recieved goal msg')
        self.goal = np.array([(msg.pose.position.x, msg.pose.position.y,0)])
        Utils.world_to_map(self.goal,self.map_info)
        self.goal = np.array((self.goal[0][0], 1300-self.goal[0][1],self.goal[0][2])).astype(int)
        #rospy.loginfo(self.goal)
        if self.environment[self.goal[1]][self.goal[0]] == 0:
            #rospy.loginfo(self.goal)

            if self.start != None and self.goal != None:
                #rospy.loginfo('getting path')
                path = np.array(self.rrt(self.environment,self.start,self.goal)).astype(float)
                #rospy.loginfo(path)
                Utils.map_to_world(path,self.map_info)
                path = path.astype(int)
                #rospy.loginfo('visualizing')
                self.visualize(path,self.start,self.goal)
        else:
            self.goal = None
            #rospy.loginfo('you clicked on a not free goal, pick another one')


    def visualize(self,path,start,goal):
        '''
        Publish various visualization messages.
        '''

        #rospy.loginfo('visualizing start')
        s = PointStamped()
        s.header = Utils.make_header("/map")
        s.point.x = start[0]
        s.point.y = start[1]
        s.point.z = 0
        self.start_pub.publish(s)

        #rospy.loginfo('visualizing goal')
        g = PointStamped()
        g.header = Utils.make_header("/map")
        g.point.x = goal[0]
        g.point.y = goal[1]
        g.point.z = 0
        self.goal_pub.publish(g)

        #rospy.loginfo('visualizing path')
        p = Path()
        p.header = Utils.make_header("/map")
        for point in path:
            pose = PoseStamped()
            pose.header = Utils.make_header("/map")
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation = Utils.angle_to_quaternion(0)
            p.poses.append(pose)
        self.path_pub.publish(p)


    def rrt(self, environment, start, goal):
        #INPUTS:
        # environment is an np.array that stores pixel information of the map (0 = free space, 1 = obstacle)
        # start and goal are the start and goal coordinates, respectively
        #OUTPUT:
        # path, a list of array coordinates coordates 
        wall = 1
        free_space = 0
        d = 1                             # tunable parameter used in steer() function
        radius = 10                         # tunable parameter used in goal test 
        start_node = SearchNode(start[:2])      # initialize start_node 
        search_tree = Tree(start_node)      # inttialize search_tree

        goal_reached = False
        #rospy.loginfo('starting rrt loop')
        while goal_reached == False:
            while True:
                rand_pos = self.sample(environment.shape[1], environment.shape[0])
                if environment[rand_pos[1]][rand_pos[0]] == free_space:
                    break
            #rospy.loginfo('got new point')
            closest_node = self.get_closest(rand_pos, search_tree.nodes)
            test_coords = self.steer(closest_node, rand_pos, d)
            #rospy.loginfo('checking for collision between nodes') 
            col = self.collision(environment, closest_node.state, test_coords)
            #rospy.loginfo(col)
            if not col:
                # FIXME -- potentially add a check for vehicle dynamics heres
                #rospy.loginfo('making new node')
                new_node = SearchNode(test_coords, closest_node)
                search_tree.add_node(new_node)

                #self.visualize()

                if self.get_distance(new_node.get_state(),goal[:2]) < radius:
                    #rospy.loginfo('extracting path from tree')
                    path = self.extract_path(start_node, new_node)
                    return path
            #rospy.loginfo('leaving nested whille loop')

    def extract_path(self, start_node, end_node):
        path = [end_node]
        while path[-1] != start_node:
            path.append(path[-1].get_parent())
        coordinates = [np.array([node.get_state()[0],1300-node.get_state()[1],0]) for node in path]
        # TODO CONVERT PIXEL TO MAP SPACE
        return np.array(coordinates[::-1])

    def get_distance(self, coords1, coords2):
        #rospy.loginfo(coords1)
        #rospy.loginfo(coords2)
        return np.linalg.norm(coords1-coords2)

    def collision(self, environment, start_point, end_point):
        # this function checks to see if the straight-ish line path between the pixels 
        # at x1,y1 and x2,y2 is clear of obstacles. It does this by using the slope of 
        # the line between the two points to obtain a set of equally-ish spaced points 
        # on that line and checking to see if all of those points are free. Note that the 
        # number of points checked is about equal to the distance between the given 
        # points but must be at least 10. 

        (x1,y1) = start_point
        (x2,y2) = end_point

        # define some useful variables such as the distance between the pixels
        dy = y2-y1
        dx = x2-x1
        d = math.floor(math.sqrt(dy**2+dx**2))

        #  if d is small, adjust it so that we check at least 10 points
        if d < 10:
            d = 10
        else:
            d = int(d)

        # check the d equally spaced points between the given points to see if 
        # to see if they are free, if one is not free return false
        for i in range(d):
            x3 = int(math.floor(x1 + dx*(i+1)/d))
            y3 = int(math.floor(y1 + dy*(i+1)/d))
            if environment[y3,x3]==1:
                return True

        # if it hasn't returned false by now, path is clear
        return False

    def get_closest(self, rand_pos, nodes):
        return nodes[np.argmin(np.array([self.get_distance(rand_pos,node.get_state()) for node in nodes]))]

    def sample(self, width, height):
        x = np.random.randint(0,width)
        y = np.random.randint(0,height)
        return np.array([x,y])

    def steer(self, node, rand_pos, d):
        node_state = node.get_state()
        return (node_state + (rand_pos-node_state)*d).astype(int)


class Tree(object):
    def __init__(self, root):
        self.root = root
        self.nodes = [root]

    def get_root(self):
        return self.root
        
    def get_nodes(self):
        return self.nodes

    def add_node(self, new_node):
        self.nodes.append(new_node)


class SearchNode(object):
    def __init__(self, state, parent_node=None):
        self.state = state
        self.parent = parent_node   
        
    def get_x(self):
        return self.state[0]
    
    def get_y(self):
        return self.state[1]

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent

if __name__=="__main__":
    rospy.init_node("rrt")
    
    stata_basement = Map("basement_fixed.png") 
    environment = stata_basement.map
    #rospy.loginfo(type(environment))
    rrt = RRT(environment)    

    #stata_basement.drawPath(path)
    rospy.spin()
