package main

import (
	"fmt"
	"log"

	. "gorgonia.org/gorgonia"
)

//标量运算
func main() {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression 全是Node
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	z, err = Add(x, y)
	if err != nil {
		log.Fatal(err)
	}

	// create a VM to run the program on
	machine := NewTapeMachine(g)

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if machine.RunAll() != nil {
		log.Fatal(err)
	}

	fmt.Printf("%v", z.Value())
	// Output: 4.5
}
