"""Paper Algorithm 1 and upstream-compatible operation/storage accounting."""
import heapq
import math

MOTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def intersects(a, b, c, d):
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def on(p, q, r):
        return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
                min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))
    x, y, z, w = cross(a,b,c), cross(a,b,d), cross(c,d,a), cross(c,d,b)
    return ((x*y < 0 and z*w < 0) or
            (x == 0 and on(a,b,c)) or (y == 0 and on(a,b,d)) or
            (z == 0 and on(c,d,a)) or (w == 0 and on(c,d,b)))


class Environment:
    def __init__(self, query, collision_backend='shapely'):
        if collision_backend not in ('shapely', 'orientation'):
            raise ValueError('Unknown collision backend')
        self.collision_backend = collision_backend
        self.query = query
        self.start, self.goal = tuple(query['start']), tuple(query['goal'])
        self.rx = tuple(query.get('range_x', query.get('x_range', ())))
        self.ry = tuple(query.get('range_y', query.get('y_range', ())))
        if len(self.rx) != 2 or len(self.ry) != 2:
            raise ValueError('range_x and range_y must contain [min, exclusive_max]')
        values = [*self.rx, *self.ry, *self.start, *self.goal]
        if len(self.start) != 2 or len(self.goal) != 2 or any(type(v) is not int for v in values):
            raise ValueError('Bounds, start and goal must be integer 2D coordinates')
        if self.rx[1]-self.rx[0] < 3 or self.ry[1]-self.ry[0] < 3:
            raise ValueError('Map must have an interior')
        self.walls = []
        for key in ('horizontal_barriers', 'vertical_barriers'):
            for barrier in query.get(key, []):
                if len(barrier) != 3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in barrier):
                    raise ValueError('Barriers must be finite coordinate triples')
                a,b,c = barrier
                self.walls.append(((b,a),(c,a)) if key.startswith('horizontal') else ((a,b),(a,c)))
        if collision_backend == 'shapely':
            from shapely.geometry import LineString
            self._line = LineString
            x0,x1 = self.rx[0],self.rx[1]-1
            y0,y1 = self.ry[0],self.ry[1]-1
            boundary = [((x,y0),(x,y1)) for x in (x0,x1)] + [((x0,y),(x1,y)) for y in (y0,y1)]
            self._segments = [LineString([a,b]) for a,b in self.walls + boundary]
        if not self.free(self.start) or not self.free(self.goal):
            raise ValueError('Start/goal lies on a barrier or outside the map interior')

    def inside(self, p):
        return self.rx[0] < p[0] < self.rx[1]-1 and self.ry[0] < p[1] < self.ry[1]-1

    def free(self, p):
        return self.inside(p) and not any(intersects(p,p,a,b) for a,b in self.walls)

    def collision(self, a, b):
        if self.collision_backend == 'shapely':
            segment = self._line([a,b])
            return any(segment.intersects(wall) for wall in self._segments)
        return not self.inside(a) or not self.inside(b) or any(intersects(a,b,c,d) for c,d in self.walls)

    def targets(self, nodes, mode):
        if mode == 'upstream':
            # Original implementation uses an extra one-cell margin and half-open obstacle sets.
            obs = set()
            for y,a,b in self.query.get('horizontal_barriers', []):
                if all(type(v) is int for v in (y,a,b)):
                    obs.update((x,y) for x in range(a,b))
            for x,a,b in self.query.get('vertical_barriers', []):
                if all(type(v) is int for v in (x,a,b)):
                    obs.update((x,y) for y in range(a,b))
            valid = [p for p in nodes if p not in obs and
                     self.rx[0]+1 < p[0] < self.rx[1]-2 and self.ry[0]+1 < p[1] < self.ry[1]-2]
        else:
            valid = [p for p in nodes if self.free(p)]
        if not valid or valid[0] != self.start:
            valid.insert(0, self.start)
        if valid[-1] != self.goal:
            valid.append(self.goal)
        if len(valid) == 1:
            valid.append(self.goal)
        return valid


def path_result(env, path, operation=None, storage=None):
    valid = bool(path) and path[0] == env.start and path[-1] == env.goal
    # Official compatibility uses segment intersections, including degenerate lines.
    # Keep the old point-based check only when auditing legacy results.
    if env.collision_backend == 'orientation':
        valid = valid and all(env.free(p) for p in path)
    valid = valid and all(not env.collision(a,b) for a,b in zip(path,path[1:]))
    return dict(path=path, valid=bool(valid), operation=operation, storage=storage,
                length=sum(math.dist(a,b) for a,b in zip(path,path[1:])) if valid else None)


def search(query, nodes=None, mode='upstream'):
    """nodes=None selects A*. A list selects LLM-A*, including empty-list fallback."""
    if mode not in ('upstream', 'paper'):
        raise ValueError('Unknown search mode')
    env = Environment(query)
    targets = env.targets(nodes,mode) if nodes is not None else None
    index = 1
    def f(p):
        return g[p] + (math.dist(p,env.goal) + (math.dist(p,targets[index]) if targets else 0))
    g, parent = {env.start:0.0}, {env.start:env.start}
    queue = [(f(env.start),env.start)]
    closed = set()
    operations = updates = 0
    found = False
    while queue:
        _,s = heapq.heappop(queue)
        if mode == 'paper' and s in closed:
            continue
        operations += 1
        closed.add(s)
        if s == env.goal:
            found = True
            break
        for dx,dy in MOTIONS:
            n = (s[0]+dx,s[1]+dy)
            if mode == 'paper' and n in closed:
                continue
            if targets and n == targets[index] and n != env.goal:
                index = min(index+1,len(targets)-1)
                queue = [(f(p),p) for _,p in queue]
                heapq.heapify(queue)
                updates += 1
            if n in closed:
                continue
            cost = math.inf if env.collision(s,n) else math.dist(s,n)
            # Include blocked neighbors in storage, as the released code does.
            if mode == 'upstream' or math.isfinite(cost):
                g.setdefault(n, math.inf)
            new = g[s] + cost
            if new < g.get(n, math.inf):
                g[n],parent[n] = new,s
                heapq.heappush(queue,(f(n),n))
    path = []
    if found:
        path = [env.goal]
        while path[-1] != env.start:
            path.append(parent[path[-1]])
        path.reverse()
    result = path_result(env,path,operations,len(g))
    result.update(target_updates=updates, unique_expanded=len(closed), search_mode=mode,
                  collision_backend=env.collision_backend)
    if targets is not None:
        result['llm_output'] = targets
    return result


class AStar:
    def __init__(self, mode='upstream'):
        self.mode = mode

    def searching(self, query, filepath=None):
        if filepath is not None:
            raise ValueError('Plotting is not implemented; omit filepath and use returned path')
        return search(query,mode=self.mode)


class LLMAStar:
    def __init__(self, model, prompt='standard', mode='upstream'):
        self.model, self.prompt, self.mode = model,prompt,mode

    def searching(self, query, filepath=None):
        from .models import parse_path
        if filepath is not None:
            raise ValueError('Plotting is not implemented; omit filepath and use returned path')
        response = self.model.generate(query,self.prompt)
        try:
            nodes = parse_path(response['text'],mode=self.mode)
            error = None
        except ValueError as exc:
            nodes,error = [],str(exc)
        result = search(query,nodes,self.mode)
        result.update(generation=response,parse_error=error)
        return result
