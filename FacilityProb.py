import numpy as np

class FacilityLocationProb:
    @staticmethod
    def determine_distances(locs1, locs2):
        diff = np.abs(locs1[:, None, :]-locs2[None, :, :])
        return diff.sum(axis=2)
    """
    Defines an instance of the capacitated facility location problem as a
    collection of numpy arrays representing important information.
    """
    def __init__(self, opening_costs, facility_locations, client_locations, capacities, demands=np.array([])):
        self.opening_costs = opening_costs
        self.facility_locations = facility_locations
        self.client_locations = client_locations
        # To satisfy triangle inequality, use grid distances :)
        self.connection_costs = self.determine_distances(facility_locations, client_locations)
        self.facility_distances = self.determine_distances(facility_locations, facility_locations)
        self.capacities = capacities
        self.num_F = len(opening_costs)
        self.num_D = len(self.connection_costs[0, :]) # facility X customer format
        if len(demands) == 0: # if no demands
            self.demands = np.ones(self.num_D)
        else:
            self.demands = demands

    def flatten(self):
        self.flat_connection_costs = self.connection_costs.flatten()



class OldFacilityLocationProb:
    """
    Main class for us. Defines an instance of the capacitated facility location problem as a
    collection of numpy arrays representing important information.
    """
    def __init__(self, opening_costs, connection_costs, capacities, demands=np.array([])):
        self.opening_costs = opening_costs
        self.connection_costs = connection_costs
        self.capacities = capacities
        self.num_F = len(opening_costs)
        self.num_D = len(connection_costs[0, :]) # facility X customer format
        if len(demands) == 0: # if no demands
            self.demands = np.ones(self.num_D)
        else:
            self.demands = demands

    def flatten(self):
        self.flat_connection_costs = self.connection_costs.flatten()