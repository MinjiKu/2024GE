# South Korea's Trade Policy in Protectionism
Economic research that takes part in International Trade <br>
We focus on empirical analysis of South Korean economic policy direction amidst changing international situation.

## Relevant Research

### Trade Wars and Trade Talks with Data
* American Economic Reviews
* Ralph Ossa, WTO Chief Economist
Empirical analysis of noncooperative and cooperative trade policy. <br>
A unified framework which nests traditional, new trade, and political economy motives for protection <br>
Can provide immediate trade policy relevance / Suggest and provide plausibility for trade policy making <br>

### Optimum Tariffs and Retaliation
* The Review of Economic Studies, Oxford University Press
* Harry G. Johnson
Theoretical Analysis <br>
Proof that country may gain by imposing a tariff, even if other countries retaliate <br>
Determine the condition under which it will gain in one special case <br>

## Research Plan and Method

### Dependencies

* matlab (.m) and python (.py)
```
pip install pandas
```

### Model

* 4 industries, 5 countries
  * ~~seaweed~~, steel, car, semiconductor (we previously included seaweed, but necessary data are missing in the industry, so we excluded)
  * Korea, USA, China, Japan, Germany
    * Chose industries where Korea has market power
    * Countries that have high trade volumes with Korea within the industries

### Optimum Tariff Calculation
calculating optimum tariff using static real datasets from all participating countries' public stats
* one-shot game
  
### Nash Equilibrium Tariff
calculating nash tariff when retaliation exist
* repeated game
* participating countries can know the next step of their couterparts

### Cooperative Tariff
calculating cooperative tariff when increased welfare is distributed evenly to all participating countries
* repeated game
* but, participants communicate among themselves
* Ossa claims that eventually trade environment switch from nash to cooperative due to decreased welfare in nash environment

### Welfare
Throughout the research, we track the change in welfare in order to compare <br>
This research aims to find tariff strategy by analyzing welfare changes

## Main problems encountered & Further endeavors needed
```calculating nash tariffs in repeating game <br> updating tariffs and using the updated tariffs for new calculation is difficult to implement <br> enaling potential room for code expansion is difficult```
