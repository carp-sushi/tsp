package main

import (
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"
)

// There are pi radians per 180 degrees.
const piRads = math.Pi / 180.0

// The radius of Earth in miles.
const radiusEarth = 3959.0

// City is the name and location (latitude and longitude) of a city.
type City struct {
	Name string
	Lat  float64
	Lon  float64
}

// Tour is a path through all cities (a possible solution).
type Tour struct {
	path []*City
}

// Genotype is the search space (the non optimized list of cities).
type Genotype struct {
	genes []*City
}

// Population is a collection of tours to optimize.
type Population struct {
	solutions []*Tour
}

// Run competing GA go-routines until either one minute has passed or an ideal score is found.
func main() {

	rand.Seed(time.Now().UnixNano())

	var wg sync.WaitGroup
	size, offspring := 1000, 100
	tours := make(chan *Tour)
	quit := make(chan int)

	// Start our GA routines
	nThreads := 8
	for i := 0; i < nThreads; i++ {
		wg.Add(1)
		go GeneticTSP(&wg, size, offspring, tours, quit)
	}

	// Listen for solutions and print the best found so far
	go func() {
		bestScore := math.MaxFloat64
		for tour := range tours {
			tourScore := tour.Score()
			if tourScore < bestScore {
				bestScore = tourScore
				fmt.Printf("Score = %f\n", tour.Score())
				tour.Print()
			}
		}
	}()

	// Run for 30 seconds
	<-time.After(30 * time.Second)
	fmt.Println("Timeout reached")
	for i := 0; i < nThreads; i++ {
		quit <- 0
	}

	// Wait for completion
	wg.Wait()
}

// Return two random values between zero and a given integer.
func randRange(n int) (int, int) {
	r0, r1 := rand.Intn(n), rand.Intn(n)
	for r0 == r1 {
		r1 = rand.Intn(n)
	}
	if r1 < r0 {
		return r1, r0
	}
	return r0, r1
}

// Great circle distance algorithm
func distance(c0, c1 *City) float64 {
	lat0, lon0 := c0.Lat, c0.Lon
	lat1, lon1 := c1.Lat, c1.Lon
	p0 := lat0 * piRads
	p1 := lat1 * piRads
	p2 := lon1*piRads - lon0*piRads
	p3 := math.Sin(p0) * math.Sin(p1)
	p4 := math.Cos(p0) * math.Cos(p1) * math.Cos(p2)
	return radiusEarth * math.Acos(p3+p4)
}

// Create a city from an array of strings.
func initCity(fields []string) (city *City, err error) {
	if len(fields) != 3 {
		err = errors.New("Invalid line format")
		return
	}
	city = &City{}
	city.Name = strings.TrimSpace(fields[0])
	city.Lat, err = strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return
	}
	city.Lon, err = strconv.ParseFloat(fields[2], 64)
	return
}

// Shuffle creates a randomized tour.
func (t *Tour) Shuffle() {
	n := len(t.path)
	for i := 0; i < n; i++ {
		r := rand.Intn(n)
		t.path[i], t.path[r] = t.path[r], t.path[i]
	}
}

// Print writes a string version of a tour to stdout.
func (t *Tour) Print() {
	for _, city := range t.path {
		fmt.Printf("%s, ", city.Name)
	}
	fmt.Printf("\n\n")
}

// Contains determines whether a city lies within a given tour.
func (t *Tour) Contains(city *City) bool {
	for _, c := range t.path {
		if c.Name == city.Name {
			return true
		}
	}
	return false
}

// Mutate is the mutation operator.
func (t *Tour) Mutate() {
	if rand.Float32() <= 0.1 {
		mn, mx := randRange(len(t.path))
		for mn < mx {
			t.path[mn], t.path[mx] = t.path[mx], t.path[mn]
			mn, mx = mn+1, mx-1
		}
	}
}

// create a new tour at random
func makeChild(path1, path2 []*City) *Tour {
	child := &Tour{}
	n := rand.Intn(len(path1))
	child.path = append(child.path, path1[:n]...)
	for _, value := range path2 {
		if !child.Contains(value) {
			child.path = append(child.path, value)
		}
	}
	child.Mutate()
	return child
}

// Crossover is the reproduction operator.
func (t *Tour) Crossover(t2 *Tour) (children []*Tour) {
	if rand.Float32() <= 0.9 {
		children = append(children, makeChild(t.path, t2.path))
		children = append(children, makeChild(t2.path, t.path))
	}
	return
}

// Score is the total distance of a tour.
func (t *Tour) Score() float64 {
	n := len(t.path) - 1
	score := distance(t.path[n], t.path[0])
	for i := 0; i < n; i++ {
		score += distance(t.path[i], t.path[i+1])
	}
	return score
}

// Init initializes the search space from file.
func (gt *Genotype) Init(file string) error {
	rawContent, err := ioutil.ReadFile(file)
	if err != nil {
		return err
	}
	content := strings.TrimSpace(string(rawContent))
	for _, line := range strings.Split(content, "\n") {
		city, err := initCity(strings.Fields(line))
		if err != nil {
			return err
		}
		gt.genes = append(gt.genes, city)
	}
	return err
}

// RandomTour creates a random tour from the search space.
func (gt *Genotype) RandomTour() *Tour {
	tour := &Tour{}
	tour.path = append(tour.path, gt.genes...)
	tour.Shuffle()
	return tour
}

// Init initializes a population of tours.
func (p *Population) Init(gt *Genotype, size int) {
	p.solutions = make([]*Tour, size)
	for i := 0; i < size; i++ {
		p.solutions[i] = gt.RandomTour()
	}
}

// Best returns the tour with the shortest path (lowest score).
func (p *Population) Best() *Tour {
	best := p.solutions[0]
	bestScore := best.Score()
	for i := 1; i < len(p.solutions); i++ {
		current := p.solutions[i]
		currentScore := current.Score()
		if currentScore < bestScore {
			best = current
			bestScore = currentScore
		}
	}
	return best
}

// Select is the selection operator.
func (p *Population) Select() (*Tour, *Tour) {
	r1, r2 := randRange(len(p.solutions))
	return p.solutions[r1], p.solutions[r2]
}

// Evolve moves the population forward a single generation.
func (p *Population) Evolve(offspring int) {
	for i := 0; i < offspring/2; i++ {
		p0, p1 := p.Select()
		for _, child := range p0.Crossover(p1) {
			i := rand.Intn(len(p.solutions))
			if child.Score() <= p.solutions[i].Score() {
				p.solutions[i] = child
			}
		}
	}
}

// GeneticTSP continually evolves a population until a 'quit' signal is received.
func GeneticTSP(wg *sync.WaitGroup, size, offspring int, tours chan *Tour, quit chan int) {
	gt := &Genotype{}
	if err := gt.Init("capitals.tsp"); err != nil {
		panic(err)
	}
	p := &Population{}
	p.Init(gt, size)
	for {
		select {
		case tours <- p.Best():
			p.Evolve(offspring)
		case <-quit:
			fmt.Println("Quit signal received")
			wg.Done()
			return
		}
	}
}
