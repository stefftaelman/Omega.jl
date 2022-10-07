# Counterfactuals

The utility of `replace` may not be obvious at first glance.
We can use `replace` and `cond` separately and in combination to ask lots of different kinds of questions.
In this example, we model the relationship between the weather outside and the reading on the thermostat inside a house.
Broadly, the model says that the weather outside is dictated by the time of day, while the temperature inside is determined by whether the air conditioning is on, and whether the window is open.

[Notebook](https://github.com/zenna/OmegaModels.jl/blob/master/models/Thermometer/Thermometer.ipynb)

First, setup simple priors over the time of day, and variables to determine whether the air conditioning is on and whether the window is open:

```julia
using Omega
timeofday = uniform([:morning, :afternoon, :evening])
is_window_open = bernoulli(0.5) # 50/50 chance of it being open
is_ac_on = bernoulli(0.3)       # only 30% chance of ac being on
```

Second, assume that the outside temperature depends on the time of day, being hottest in the afternoon, but cold at night:

```julia
function outside_temp_(rng)
  if timeofday(rng) == :morning
    normal(rng, 20.0, 1.0)
  elseif timeofday(rng) == :afternoon
    normal(rng, 32.0, 1.0)
  else
    normal(rng, 10.0, 1.0)
  end
end
```

In this style we have to use  `ciid` to convert a function into a `RandVar`. For more elaboration, check out `? ciid`.

```julia
outside_temp = ciid(outside_temp_)
```

The `inside_temp` before considering the effects of the window is room temperature, unless the ac is on, which makes it colder.

```julia
function inside_temp_(rng)
  if Bool(is_ac_on(rng))
    normal(rng, 20.0, 1.0)
  else
    normal(rng, 25.0, 1.0)
  end
end

inside_temp = ciid(inside_temp_)
```

Finally, the thermostat reading is `inside_temp` if the window is closed (we have perfect insulation), otherwise it's just the average of the outside and inside temperature

```julia
function thermostat_(rng)
  if Bool(is_window_open(rng))
    (outside_temp(rng) + inside_temp(rng)) / 2.0
  else
    inside_temp(rng)
  end
end

thermostat = ciid(thermostat_)
```

Now with the model built, we can ask some questions:

### Samples from the prior
The simplest task is to sample from the prior, _i.e._, simulate some input states to the model:

```julia
julia> rand((timeofday, is_window_open, is_ac_on, outside_temp, inside_temp, thermostat), 5, alg = RejectionSample)
5-element Array{Any,1}:
 (:afternoon, 0.0, 0.0, 32.349, 26.441, 26.441)   
 (:afternoon, 1.0, 0.0, 30.751, 25.143, 27.947)
 (:morning, 1.0, 0.0, 16.928, 24.146, 20.537)     
 (:afternoon, 1.0, 0.0, 30.521, 25.370, 27.946)
 (:morning, 1.0, 1.0, 16.495, 20.203, 18.349) 
```

### Conditional Inference
- You enter the room and the thermostat reads hot - let's say above 30 degrees. what does this tell you about the variables?

```julia
julia> samples = rand((timeofday, is_window_open, is_ac_on, outside_temp, inside_temp, thermostat), thermostat > 30.0, 5, alg = RejectionSample)
5-element Array{Any,1}:
 (:evening, 1.0, 0.0, 33.64609872046609, 26.822449458789542, 30.234274089627817) 
 (:afternoon, 1.0, 0.0, 34.37763909867243, 26.16221853550574, 30.269928817089088)
 (:evening, 1.0, 0.0, 34.32198183192978, 26.6773921624331, 30.499686997181442)   
 (:afternoon, 1.0, 0.0, 34.05126597960254, 26.51833791813246, 30.2848019488675)  
 (:afternoon, 1.0, 0.0, 32.92982568498735, 27.56800059609554, 30.248913140541447)
```

## Counterfactuals
Asking counterfactual questions means that we examine the outcome of a scenario in a parallel world where we had done something different. For instance:

- If we had closed the window and turned on the AC would that have made it hotter or colder?

```julia
thermostatnew = replace(thermostat, is_ac_on => 1.0, is_window_open => 0.0)
```

So, if we generate some new samples from this different situation where we've intervened on the the AC and window, do we see that it (on average) gets hotter or colder?

```julia
diffsamples = rand(thermostatnew - thermostat, 10000, alg = RejectionSample)
mean(diffsamples)
```

The intervention will make the thermostat colder.  But we can look more closely at the distribution:

```
julia> UnicodePlots.histogram([diffsamples...])
                  ┌                                        ┐ 
   [-12.0, -10.0) ┤ 17                                       
   [-10.0,  -8.0) ┤▇▇▇▇▇▇▇▇ 862                              
   [ -8.0,  -6.0) ┤▇▇▇▇▇ 541                                 
   [ -6.0,  -4.0) ┤▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 3844   
   [ -4.0,  -2.0) ┤▇▇▇▇▇▇▇▇ 870                              
   [ -2.0,   0.0) ┤▇▇▇▇▇ 519                                 
   [  0.0,   2.0) ┤▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 1970                    
   [  2.0,   4.0) ┤▇▇▇▇▇▇▇▇ 907                              
   [  4.0,   6.0) ┤▇▇▇▇ 439                                  
   [  6.0,   8.0) ┤ 31                                       
                  └                                        ┘ 
                                  Frequency
```

- In what scenarios would it still be hotter after turning on the AC and closing the window?

```julia
rand((timeofday, outside_temp, inside_temp, thermostat), thermostatnew - thermostat > 0.0, 10, alg = RejectionSample)
```

We see that in the evenings and mornings when it is cold outside, we'd have made it hotter inside by closing the window, even if the AC is on.