from statistics import mean, stdev


class TimeLogger:
    def __init__(self) -> None:
        self.fitness_time = []
        self.is_fitness_time_agg = False
        self.epoch_time = []
        self.is_epoch_time_agg = False
        self.criterion_time = dict()
        self.is_criterion_time_agg = False

    def add_fitness_time(self, fitness_time):
        if not self.is_fitness_time_agg:
            self.fitness_time.append(fitness_time)

    def add_epoch_time(self, epoch_time):
        if not self.is_epoch_time_agg:
            self.epoch_time.append(epoch_time)

    def add_criterion_time(self, criterion_name, criterion_time):
        if not self.is_criterion_time_agg:
            if criterion_name in self.criterion_time:
                self.criterion_time.get(criterion_name).append(criterion_time)
            else:
                self.criterion_time.update({criterion_name: [criterion_time]})

    def agg_fitness_time(self):
        last_fit_time = self.fitness_time
        self.fitness_time = {'fit_min': min(last_fit_time),
                             'fit_avg': mean(last_fit_time),
                             'fit_max': max(last_fit_time),
                             'fit_std': stdev(last_fit_time)}
        self.is_fitness_time_agg = True
        return self.fitness_time

    def agg_epoch_time(self):
        last_epoch_time = self.epoch_time
        self.epoch_time = {'epoch_min': min(last_epoch_time),
                           'epoch_avg': mean(last_epoch_time),
                           'epoch_max': max(last_epoch_time),
                           'epoch_std': stdev(last_epoch_time)}
        self.is_epoch_time_agg = True
        return self.epoch_time

    def agg_criterion_time(self):
        for criterion in self.criterion_time.keys():
            last_criterion_time = self.criterion_time.get(criterion)
            self.criterion_time.update({
                criterion: {
                        f'crit_{criterion}_min': min(last_criterion_time),
                        f'crit_{criterion}_avg': mean(last_criterion_time),
                        f'crit_{criterion}_max': max(last_criterion_time),
                        f'crit_{criterion}_std': stdev(last_criterion_time)}})
        self.is_criterion_time_agg = True
        return self.criterion_time

    def get_fitness_time(self):
        return self.fitness_time

    def get_epoch_time(self):
        return self.epoch_time

    def get_criterion_time(self):
        return self.criterion_time
