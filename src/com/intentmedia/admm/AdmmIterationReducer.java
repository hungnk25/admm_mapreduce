package com.intentmedia.admm;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.logging.Logger;

import static com.intentmedia.admm.AdmmIterationHelper.admmMapperContextToJson;
import static com.intentmedia.admm.AdmmIterationHelper.mapToJson;

import gurobi.*;


public class AdmmIterationReducer extends MapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {

    private static final Logger LOG = Logger.getLogger(AdmmIterationReducer.class.getName());
    private static final IntWritable ZERO = new IntWritable(0);
    private Map<String, String> outputMap = new HashMap<String, String>();
    private int iteration;
    private int numberOfMappers;

    @Override
    public void configure(JobConf job) {
        super.configure(job);
        iteration = Integer.parseInt(job.get("iteration.number"));
        numberOfMappers = job.getNumMapTasks();
        numberOfMappers = 9;
    }

    @Override
    public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter)
            throws IOException {

        AdmmReducerContextGroup context = new AdmmReducerContextGroup(values, numberOfMappers, LOG, iteration);
        localReducerOptimization(context); // solve local optimization problem at Reducer

        if(outputMap.size() > 0) {
            output.collect(ZERO, new Text(mapToJson(outputMap))); // emit key-value here, or write to HDFS output data

            
        }
    }

    private void localReducerOptimization(AdmmReducerContextGroup context) throws IOException {

        String[] splitIds = context.getSplitIds();
        
        // solve local optimization problem at reducer here
		int N = 9; // number of MG
		
		double gamma = 0.01/2;

		double[] UpdatedY0 = new double[N];
		double[][] UpdatedX0 = new double[N][N];
		double[][] LambdaGroup = context.getLambdaGroup();
		double[][][] MuGroup = context.getMuGroup();
		double[][] Y0kGroup = context.getY0kGroup();
		double[][][] X0kGroup = context.getX0kGroup();
        
        try {


			
			double Gen_min = 10;
			double Gen_max = 126.33;
			double FL_max[][] = new double [][]{
					{5000,	0,	0,	10,	0,	0,	0,	0, 0},
					{0,	5000,	0,	0,	0,	0,	0,	10,	0},
					{0,	0,	5000,	0,	0,	10,	0,	0,	0},
					{10,	0,	0,	5000,	10,	0, 0,	0,10},
					{0,	0,	0,	10,	5000,	10,	0,	0,	0},
					{0,	0,	10,	0,	10,	5000,	10,	0,	0},
					{0,	0,	0,	0,	0,	10,	5000,	10,	0},
					{0,	10,	0,	0,	0,	0,	10,	5000,	10},
					{0,	0,	0,	10,	0,	0,	0,	10,	5000}			
														};

  	      
			double FL0_max[] = new double[]{ 53.67,53.67,53.67,53.67,53.67,53.67,53.67,53.67,53.67 };
			double Demand[] = new double[] {108,97,180,74,71,136,125,171,175};
			double C_main[] = new double[] {0.01, 13.5};
			double C_MG[][] = new double [][]{ 
					{0.0222, 20},
					{0.1177, 40},
					{0.0455, 20},
					{0.0319, 40},
					{0.4286, 40},
					{0.5263, 20},
					{0.0490, 20},
					{0.2083, 20},
					{0.0645, 40},
								};
			double Delta_MG[]  = new double [] { 25.2667,25.2667,25.2667,25.2667,25.2667,25.2667,25.2667,25.2667,25.2667 };
			double[] one_matrix = new double[]{1,1,1,1,1,1,1,1,1};

  	      // Model
  	      GRBEnv env = new GRBEnv();
  	      GRBModel model = new GRBModel(env);
  	      model.set(GRB.StringAttr.ModelName, "Problem");

  	      // variable for mg generation in normal mode
  	      GRBVar[][] X0 = new GRBVar[N][N];
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  for (int j = 0; j < N; j++)
  	    	  {
  	    		  X0[i][j] = model.addVar(0, FL_max[i][j], 1, GRB.CONTINUOUS, "Xnormal" + i + "." + j);
  	    	  }
  	    	  
  	      }
  	      // variable for main grid generation in normal mode
  	      GRBVar[] Y0 = new GRBVar[N];
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  Y0[i] = model.addVar(0,FL0_max[i], 1, GRB.CONTINUOUS, "Ynormal" + i);  	    	
  	      }
  	        	      
  	      	      
  	        	     
  	      // slack variable for total generation mai grid
  	      GRBVar Y0_gen = model.addVar(0, 5000, 1, GRB.CONTINUOUS, "Y0_gen");
  	      GRBVar[] X0_gen = new GRBVar[N];
  	      for (int i = 0; i < N; i++)
  	      {
  	    	X0_gen[i] = model.addVar(Gen_min,Gen_max,1,GRB.CONTINUOUS,"X0_gen"+i);
  	      }
  	      
  	      GRBVar[][] Y0_deviation = new GRBVar[N][N];
  	      for (int i = 0; i < N; i++){
  	    	  for(int j = 0; j < N; j++){
  	    		Y0_deviation[i][j] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 1, GRB.CONTINUOUS,"Y0_deviation" + i + "." + j);	
  	    	  }
  	      }
  	      
  	    	
  	      GRBVar[][][] X0_deviation = new GRBVar[N][N][N];
  	      for (int i = 0; i < N; i++){
  	    	  for(int j = 0; j < N; j++){
  	    		  for(int k = 0; k < N; k++){
  	    			X0_deviation[i][j][k] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 1, GRB.CONTINUOUS,"Y0_deviation" + i + "." + j + "." + k);  
  	    		  }	
  	    	  }
  	      }

  	      
  	      model.update();
  	      
  	      //set objective function
  	      
  	      // cost for main grid generation

  	      GRBQuadExpr MainGrid_cost = new GRBQuadExpr();
  	      MainGrid_cost.addTerm(C_main[0],Y0_gen,Y0_gen);
  	      MainGrid_cost.addTerm(C_main[1],Y0_gen);

  	      
  	      // cost for microgird generation
  	      GRBQuadExpr MG_cost = new GRBQuadExpr();
  	      
  	      for(int i = 0; i < N; i++) // loop for all microgrid
  	      {
  	    	  MG_cost.addTerm(C_MG[i][0], X0_gen[i],X0_gen[i]);
  	    	  MG_cost.addTerm(C_MG[i][1],X0_gen[i]);
  	    	  
  	      }
  	      
  	      
  	      GRBQuadExpr PenYcost = new GRBQuadExpr();
  	      for(int i = 0; i < N; i++){
  	    	PenYcost.addTerms(one_matrix,Y0_deviation[i],Y0_deviation[i]);
  	      }
  	      
  	      
  	      
  	      
  	      GRBQuadExpr PenXcost = new GRBQuadExpr();
  	      for(int i = 0; i < N; i++){
  	    	  for(int j = 0; j < N; j++){
  	    		  PenXcost.addTerms(one_matrix, X0_deviation[i][j], X0_deviation[i][j]);
  	    	  }
  	      }
  	      
  	      GRBLinExpr LamYcost = new GRBLinExpr();
  	      for(int i = 0; i < N; i++){
  	    	  LamYcost.addTerms(LambdaGroup[i], Y0);
  	      }
  	      
  	      GRBLinExpr MuXcost = new GRBLinExpr();
  	      for(int i = 0; i < N; i++){
  	    	  for(int j = 0; j < N; j++){
  	    		  for(int k = 0; k < N; k++){
  	    			  MuXcost.addTerm(MuGroup[i][j][k], X0[j][k]);
  	    		  }
  	    	  }
  	      }


  	      
  	      GRBQuadExpr Total_cost = new GRBQuadExpr();
  	      Total_cost.add(MainGrid_cost);
  	      Total_cost.add(MG_cost);
  	      Total_cost.multAdd(-1, LamYcost);
  	      Total_cost.multAdd(-1, MuXcost);
  	      Total_cost.multAdd(gamma, PenYcost);
  	      Total_cost.multAdd(gamma, PenXcost);
  	      
  	      //set objective function
  	      model.setObjective(Total_cost,GRB.MINIMIZE);
  	      
  	      // Construct constraints for the problem
  	      
  	      
  	      GRBLinExpr Yterm = new GRBLinExpr();
  	      for(int i = 0; i < N; i++){
  	    	  for(int j = 0; j < N; j++){
  	    		  Yterm = new GRBLinExpr();
  	    		  Yterm.addTerm(-1, Y0[i]);
  	    		  Yterm.addConstant(Y0kGroup[i][j]);
  	    		  model.addConstr(Y0_deviation[i][j], GRB.EQUAL, Yterm,"Yslack" + i + "." + j);
  	    	  }
  	      }
  	      
  	      GRBLinExpr Xterm = new GRBLinExpr();
  	      for(int i =0; i < N; i++){
  	    	  for(int j = 0; j < N; j++){
  	    		  for(int k = 0; k < N; k++){
  	    			  Xterm = new GRBLinExpr();
  	    			  Xterm.addTerm(-1, X0[j][k]);
  	    			  Xterm.addConstant(X0kGroup[i][j][k]);
  	    			  model.addConstr(X0_deviation[i][j][k], GRB.EQUAL, Xterm, "Xslack" + i + "." + j + "."+ k);
  	    		  }
  	    	  }
  	      }

  	      
  	      // constraint for slack variables Y0_gen and X_0 gen
  	      GRBLinExpr Main_gen = new GRBLinExpr();
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  Main_gen.addTerm(1, Y0[i]);
  	      }
  	      model.addConstr(Y0_gen,GRB.EQUAL, Main_gen, "MainGen");
  	      
  	    GRBLinExpr MG_gen = new GRBLinExpr();
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  MG_gen = new GRBLinExpr();
  	    	  for (int j = 0; j < N; j++)
  	    	  {
  	    		  MG_gen.addTerm(1, X0[j][i]);
  	    	  }
  	    	  model.addConstr(X0_gen[i], GRB.EQUAL, MG_gen, "MG gen" + i);
  	      }
  	      
  	      // constraint for normal operation mode
	    	GRBLinExpr Demand_const = new GRBLinExpr();

  	      for (int i = 0; i < N; i++)
  	      {
  	    	Demand_const = new GRBLinExpr();
  	    	for (int j = 0; j < N; j++)
  	    	{
  	    		Demand_const.addTerm(1,X0[i][j]);
  	    		
  	    	}
  	    	Demand_const.addTerm(1, Y0[i]);
  	    	model.addConstr(Demand_const, GRB.EQUAL, Demand[i], "MG Demand" + i);
  	      }
  	      
  	      
  	      
  	       	      
  	      
  	    model.optimize();
  	    
  	    for(int i = 0; i < N; i++){
  	    	UpdatedY0[i] = Y0[i].get(GRB.DoubleAttr.X);
  	    }
  	    
  	    for(int i = 0; i < N; i++){
  	    	for(int j = 0; j < N; j++){
  	    		UpdatedX0[i][j] = X0[i][j].get(GRB.DoubleAttr.X);
  	    	}
  	    }
  	    
  	    // Dispose of model and environment

        model.dispose();
        env.dispose();
  	      
  	      
	    } catch (GRBException e) {
	        System.out.println("Error code: " + e.getErrorCode() + ". " +
	            e.getMessage());
	      }
        
        
        ///

        for (int mapperNumber = 0; mapperNumber < context.getNumberOfMappers(); mapperNumber++) {
            AdmmMapperContext admmMapperContext =
                    new AdmmMapperContext(LambdaGroup[mapperNumber], MuGroup[mapperNumber], UpdatedY0,UpdatedX0);
            String currentSplitId = splitIds[mapperNumber];
            outputMap.put(currentSplitId, admmMapperContextToJson(admmMapperContext));
            LOG.info(String.format("Iteration %d Reducer Setting splitID %s", iteration, currentSplitId));
        }
    }

    
    public static enum IterationCounter {
        ITERATION
    }
}
