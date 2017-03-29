package com.intentmedia.admm;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.*;

import java.io.IOException;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import static com.intentmedia.admm.AdmmIterationHelper.*;
import gurobi.*;

public class AdmmIterationMapper extends MapReduceBase
        implements Mapper<LongWritable, Text, IntWritable, Text> {

    private static final IntWritable ZERO = new IntWritable(0);
    private static final Logger LOG = Logger.getLogger(AdmmIterationMapper.class.getName());

    private int iteration;
    private FileSystem fs;
    private Map<String, String> splitToParameters;

    private String previousIntermediateOutputLocation;
    private Path previousIntermediateOutputLocationPath;
    //private Path new_previousIntermediateOutputLocationPath;

    @Override
    public void configure(JobConf job) {
        iteration = Integer.parseInt(job.get("iteration.number"));
        previousIntermediateOutputLocation = job.get("previous.intermediate.output.location");
        previousIntermediateOutputLocationPath = new Path(previousIntermediateOutputLocation);
        
        //new_previousIntermediateOutputLocationPath = new Path(previousIntermediateOutputLocation);
        
        try {
            fs = FileSystem.get(job);
        }
        catch (IOException e) {
            LOG.log(Level.FINE, e.toString());
        }

        splitToParameters = getSplitParameters();
    }

    protected Map<String, String> getSplitParameters() {
        return readParametersFromHdfs(fs, previousIntermediateOutputLocationPath, iteration);
    }

    @Override
    public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter)
            throws IOException {
        FileSplit split = (FileSplit) reporter.getInputSplit();
        String splitId = key.get() + "@" + split.getPath();
        splitId = removeIpFromHdfsFileName(splitId); // this ID is used to determine which subproblem is


        AdmmMapperContext mapperContext;
        if (iteration == 0) {
            mapperContext = new AdmmMapperContext();
        }
        else {
            mapperContext = assembleMapperContextFromCache(splitId);
        }
        AdmmReducerContext reducerContext = localMapperOptimization(mapperContext, Integer.parseInt(value.toString())); // solve the sub-problem in each mapper here

        LOG.info(String.format("Iteration %d Mapper outputting splitId %s", iteration, splitId));
        output.collect(ZERO, new Text(splitId + "::" + admmReducerContextToJson(reducerContext))); // emit key-value here
    }


// this function 
    private AdmmMapperContext assembleMapperContextFromCache( String splitId) throws IOException {
        if (splitToParameters.containsKey(splitId)) {
            AdmmMapperContext preContext = jsonToAdmmMapperContext(splitToParameters.get(splitId));
            return new AdmmMapperContext(
            		preContext.getLambda(),
            		preContext.getMu(),
            		preContext.getY0normal(),
                    preContext.getX0normal());
        }
        else {
            LOG.log(Level.FINE, String.format("Key not found. Split ID: %s Split Map: %s", splitId, splitToParameters.toString()));
            throw new IOException(String.format("Key not found.  Split ID: %s Split Map: %s", splitId, splitToParameters.toString()));
        }
    }
    
    
    private AdmmReducerContext localMapperOptimization(AdmmMapperContext context, int MGislanded) {
        
        // solve local optimization at each mapper here and return a reducer context to emit to the reducer
    	
		int N = 9; // number of MG
		double tau = 10;
		double gamma = 0.01/2;
		double[] UpdatedLambda = new double[N];
		double[][] UpdatedMu = new double[N][N];
		double[] UpdatedY0k = new double[N];
		double[][] UpdatedX0k = new double[N][N];
		
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
			
			double Delta_MG[]  = new double [] { 25.2667,25.2667,25.2667,25.2667,25.2667,25.2667,25.2667,25.2667,25.2667 };
			double[] paraLambda = context.getLambda();
			double[][] paraMu = context.getMu();
			double[] paraYnormal = context.getY0normal();
			double[][] paraXnormal = context.getX0normal();
			double[] one_matrix = new double[]{1,1,1,1,1,1,1,1,1};
			
			// variables for calculating new Lambda, Mu

  	      // Model
  	      GRBEnv env = new GRBEnv();
  	      GRBModel model = new GRBModel(env);
  	      model.set(GRB.StringAttr.ModelName, "Islanded Problem" + MGislanded);

  	      // define all variables 
  	      GRBVar[][] Xk = new GRBVar[N][N];
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  for (int j = 0; j < N; j++)
  	    	  {
  	    		  Xk[i][j] = model.addVar(0, FL_max[i][j], 1, GRB.CONTINUOUS, "Xnormal" + i + "." + j);
  	    	  }
  	    	  
  	      }
  	      // variable for main grid generation in normal mode
  	      GRBVar[] Yk = new GRBVar[N];
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  Yk[i] = model.addVar(0,FL0_max[i], 1, GRB.CONTINUOUS, "Ynormal" + i);  	    	
  	      }
  	      
  	      // variable for islanded 
  	      
  	      GRBVar[][] X0k = new GRBVar[N][N];
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  for (int j = 0; j < N; j++)
  	    	  {  	    		  
  	  	    	X0k[i][j] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 1, GRB.CONTINUOUS, "Xislanded" + i + "." + j );  	    		  
  	    	  }
  	    	  
  	      }
  	      
  	      // variable for Y islanded
  	      GRBVar[] Y0k = new GRBVar[N];
  	      for (int i = 0; i < N; i++)
  	      {  	    	 
  	    	  Y0k[i] = model.addVar(-GRB.INFINITY, GRB.INFINITY, 1, GRB.CONTINUOUS, "Yislanded" + i );  	    	  
  	    	  
  	      }
  	      
  	      // variable for Zk
  	      GRBVar Zk = model.addVar(0, 1, 1, GRB.CONTINUOUS, "Zcut");
  	      
  	      // slack variable for penalty Terms  	      
  	     	      
  	      GRBVar[] Y0_deviation = new GRBVar[N];
  	      for(int i=0;i<N;i++){
  	    	  Y0_deviation[i]= model.addVar(-GRB.INFINITY, GRB.INFINITY, 1, GRB.CONTINUOUS,"Y0_deviation" + i);
  	      }
  	      
  	      // slack variable for X penalty terms
  	      GRBVar[][] X0_deviation = new GRBVar[N][N];
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  for (int j = 0; j < N; j++){  		  
  	    	  		X0_deviation[i][j] = model.addVar(-GRB.INFINITY,GRB.INFINITY,1,GRB.CONTINUOUS,"X0_deviation" + i + "." + j);
  	    	  }
  	      }
  	      
  	      
  	      
  	      model.update();
  	      
  	      //set objective function
  	      
  	      GRBLinExpr Curtail_cost = new GRBLinExpr();
  	      Curtail_cost.addTerm(Demand[MGislanded-1], Zk);
  	      
  	      // cost for Lambda*Y0k term
  	      
  	      GRBLinExpr LambdaY0k_Term = new GRBLinExpr();
  	      LambdaY0k_Term.addTerms(paraLambda, Y0k);
  	      
  	      // cost for Mu*X0k term
  	      
  	      GRBLinExpr MuX0k_Term = new GRBLinExpr();
  	     for (int i = 0; i < N; i++){
  	    	 MuX0k_Term.addTerms(paraMu[i], Xk[i]);
  	     }
  	     
  	     // cost for penalty term Y
  	     GRBQuadExpr penaltyY = new GRBQuadExpr();
  	     penaltyY.addTerms(one_matrix, Y0_deviation, Y0_deviation);
  	     
  	     
  	     // cost for penalty term X
  	     GRBQuadExpr penaltyX = new GRBQuadExpr();
  	     for(int i = 0; i < N; i++){
  	    	 penaltyX.addTerms(one_matrix, X0_deviation[i],X0_deviation[i]);
  	     }

 	      
  	       
  	      // Total cost :   	      	      
  	      GRBQuadExpr Total_cost = new GRBQuadExpr();
  	      Total_cost.multAdd(tau,Curtail_cost);
  	      Total_cost.multAdd(1, LambdaY0k_Term);
  	      Total_cost.multAdd(1, MuX0k_Term);
  	      Total_cost.multAdd(gamma, penaltyY);
  	      Total_cost.multAdd(gamma, penaltyX);
  	      
  	      //set objective function
  	      model.setObjective(Total_cost,GRB.MINIMIZE);
  	      
  	      // Construct constraints for the problem
  	      
  	      // constraint for slack variables Y0_deviation and X_0deviation
  	      GRBLinExpr PenaltyY_slack = new GRBLinExpr();
  	      for (int i = 0; i < N; i++)
  	      {
  	    	  PenaltyY_slack = new GRBLinExpr();
  	    	  PenaltyY_slack.addTerm(1, Y0_deviation[i]);
  	    	  PenaltyY_slack.addConstant(paraYnormal[i]);
  	    	  model.addConstr(Y0k[i], GRB.EQUAL, PenaltyY_slack, "Y0_deviation" + i);
  	      }
  	      
  	      GRBLinExpr PenaltyXslack = new GRBLinExpr();
  	      for(int i = 0; i < N; i++){
  	    	  for(int j = 0; j < N; j++){
  	    		  PenaltyXslack = new GRBLinExpr();
  	    		  PenaltyXslack.addTerm(1, X0_deviation[i][j]);
  	    		  PenaltyXslack.addConstant(paraXnormal[i][j]);
  	    		  model.addConstr(X0k[i][j], GRB.EQUAL, PenaltyXslack,"X0_deviation" + i + "." + j);
  	    	  }
  	      }
  	      
  	      GRBLinExpr Load_satisfied = new GRBLinExpr();
  	      for(int i=0; i < N; i++){
  	    	  if(i == MGislanded){
  	    		  model.addConstr(Yk[i], GRB.EQUAL, 0, "Yk islanded" + i);
  	    		  Load_satisfied = new GRBLinExpr();
  	    		  Load_satisfied.addTerm(Demand[i], Zk);
  	    		  Load_satisfied.addTerms(one_matrix, Xk[i]);
  	    		  model.addConstr(Load_satisfied, GRB.GREATER_EQUAL, Demand[i], "Demand constraint" + i);
  	    	  }
  	    	  else{
  	    		  Load_satisfied = new GRBLinExpr();
  	    		  Load_satisfied.addTerms(one_matrix, X0k[i]);
  	    		  Load_satisfied.addTerm(1,Yk[i]);
  	    		  model.addConstr(Load_satisfied, GRB.EQUAL, Demand[i], "Demand constraint" + i);
  	    	  }
  	      }
  	      
  	    
  	      // ramping constraint
  	      GRBLinExpr GenIslanded = new GRBLinExpr();
  	      GRBLinExpr GenNormal = new GRBLinExpr();
  	      GRBLinExpr Ramp = new GRBLinExpr();
  	      
  	      for (int i = 0; i < N; i++){ // for each user
  	    	  GenIslanded = new GRBLinExpr();
  	    	  GenNormal = new GRBLinExpr();
  	    	  for(int j = 0; j < N; j++){
  	    		  GenIslanded.addTerm(1,Xk[j][i]);
  	    		  GenNormal.addTerm(1, X0k[j][i]);
  	    	  }  
  	    	  model.addConstr(GenNormal, GRB.LESS_EQUAL , Gen_max , "Generation normal max" + i);
  	    	  model.addConstr(GenNormal, GRB.GREATER_EQUAL, Gen_min, "Genration normal min" + i);
  	    	  model.addConstr(GenIslanded, GRB.LESS_EQUAL, Gen_max, "Generation islanded max" + i);
  	    	  model.addConstr(GenIslanded, GRB.GREATER_EQUAL, Gen_min, "Generation islaned min" + i);
  	    	  
  	    	  Ramp.add(GenIslanded);
  	    	  Ramp.multAdd(-1, GenNormal);
  	    	  model.addConstr(Ramp, GRB.LESS_EQUAL, Delta_MG[i], "Ramping max" + i);
  	    	  Ramp.addConstant(Delta_MG[i]);
  	    	  model.addConstr(Ramp,  GRB.GREATER_EQUAL,  0,  "Ramping min" + i);
  	    	  
  	      }   

  	    model.optimize();
  	    
  	    // update dual variable
  	    for(int i = 0; i < N; i++){
  	    	UpdatedLambda[i] = paraLambda[i] + 2*gamma*(Y0k[i].get(GRB.DoubleAttr.X) - paraYnormal[i]);
  	    	UpdatedY0k[i] = Y0k[i].get(GRB.DoubleAttr.X);
  	    }
  	    
  	    for(int i = 0; i < N; i++){
  	    	for(int j = 0; j < N; j++){
  	    		UpdatedMu[i][j] = paraMu[i][j] + 2*gamma*( X0k[i][j].get(GRB.DoubleAttr.X) - paraXnormal[i][j] );
  	    		UpdatedX0k[i][j] = X0k[i][j].get(GRB.DoubleAttr.X);
  	    	}
  	    }
  	   // Dispose of model and environment
  	    
        model.dispose();
        env.dispose();
  	      
  	      
	    } catch (GRBException e) {
	        System.out.println("Error code: " + e.getErrorCode() + ". " +
	            e.getMessage());
	      }
    	

    	
     return new AdmmReducerContext(UpdatedLambda, UpdatedMu, UpdatedY0k,UpdatedX0k);   
    }
    
    
    
}