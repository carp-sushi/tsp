package main

import (
	"bufio"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
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
	path []City
}

// Genotype is the search space (the non optimized list of cities).
type Genotype struct {
	genes []City
}

// Population is a collection of tours to optimize.
type Population struct {
	solutions []Tour
}

// Run competing GA go-routines to find a "good enough" solution to the TSP.
func main() {
	var wg sync.WaitGroup
	size, offspring := 100, 10
	tours := make(chan Tour)
	quit := make(chan int)

	// Start our GA routines
	for range max(2, runtime.NumCPU()/2+1) {
		wg.Add(1)
		go GeneticTSP(&wg, size, offspring, tours, quit)
	}

	// Collect solutions and print the best found
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

	// Terminates TSP go-routines after 10 seconds
	go func() {
		<-time.After(10 * time.Second)
		close(quit)
	}()

	// Wait for completion
	wg.Wait()
	close(tours)
	fmt.Println("Done.")
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
func distance(c0, c1 City) float64 {
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
func initCity(fields []string) (city City, err error) {
	if len(fields) != 3 {
		err = errors.New("Invalid line format")
		return
	}
	name := strings.TrimSpace(fields[0])
	var lat float64
	lat, err = strconv.ParseFloat(fields[1], 64)
	if err != nil {
		return
	}
	var lon float64
	lon, err = strconv.ParseFloat(fields[2], 64)
	if err != nil {
		return
	}
	return City{name, lat, lon}, nil
}

// Copy clones a city
func (c City) Copy() City {
	return City{c.Name, c.Lat, c.Lon}
}

// Shuffle creates a randomized tour.
func (t *Tour) Shuffle() {
	rand.Shuffle(len(t.path), func(i, j int) {
		t.path[i], t.path[j] = t.path[j], t.path[i]
	})
}

// Print writes a string version of a tour to stdout.
func (t Tour) Print() {
	for _, city := range t.path {
		fmt.Printf("%s, ", city.Name)
	}
	fmt.Printf("\n\n")
}

// Contains determines whether a city lies within a given tour.
func (t Tour) Contains(city City) bool {
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
func makeChild(path1, path2 []City) (child Tour) {
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
func (t Tour) Crossover(t2 Tour) (children []Tour) {
	if rand.Float32() <= 0.9 {
		children = append(children, makeChild(t.path, t2.path))
		children = append(children, makeChild(t2.path, t.path))
	}
	return
}

// Score is the total distance of a tour.
func (t Tour) Score() float64 {
	n := len(t.path) - 1
	score := distance(t.path[n], t.path[0])
	for i := range n {
		score += distance(t.path[i], t.path[i+1])
	}
	return score
}

// Init initializes the search space from file.
func (gt *Genotype) Init(file string) error {
	reader, err := os.Open(file)
	if err != nil {
		return err
	}
	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		city, err := initCity(strings.Fields(scanner.Text()))
		if err != nil {
			return err
		}
		gt.genes = append(gt.genes, city)
	}
	return nil
}

// RandomTour creates a random tour from the search space.
func (gt Genotype) RandomTour() (tour Tour) {
	tour.path = make([]City, len(gt.genes))
	for i, gene := range gt.genes {
		tour.path[i] = gene.Copy()
	}
	tour.Shuffle()
	return
}

// Init initializes a population of tours.
func (p *Population) Init(gt Genotype, size int) {
	p.solutions = make([]Tour, size)
	for i := range size {
		p.solutions[i] = gt.RandomTour()
	}
}

// Best returns the tour with the shortest path (lowest score).
func (p Population) Best() (best Tour) {
	bestScore := math.MaxFloat64
	for _, current := range p.solutions {
		currentScore := current.Score()
		if currentScore < bestScore {
			best = current
			bestScore = currentScore
		}
	}
	return
}

// Select is the selection operator.
func (p Population) Select() (Tour, Tour) {
	r1, r2 := randRange(len(p.solutions))
	return p.solutions[r1], p.solutions[r2]
}

// Evolve moves the population forward a single generation.
func (p *Population) Evolve(offspring int) {
	for range offspring / 2 {
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
func GeneticTSP(wg *sync.WaitGroup, size, offspring int, tours chan Tour, quit chan int) {
	gt := Genotype{}
	if err := gt.Init("capitals.tsp"); err != nil {
		panic(err)
	}
	p := Population{}
	p.Init(gt, size)
	for {
		select {
		case tours <- p.Best():
			p.Evolve(offspring)
		case <-quit:
			wg.Done()
			return
		}
	}
}
