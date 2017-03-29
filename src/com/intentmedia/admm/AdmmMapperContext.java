package com.intentmedia.admm;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.codehaus.jackson.annotate.JsonProperty;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import static com.intentmedia.admm.AdmmIterationHelper.admmMapperContextToJson;
import static com.intentmedia.admm.AdmmIterationHelper.jsonToAdmmMapperContext;

public class AdmmMapperContext implements Writable {

    private static final int N = 9;
    
    @JsonProperty("Lambda")
    private double[] Lambda; // dual variable from previous iteration
    
    @JsonProperty("Mu")
    private double[][] Mu; // dual variable from previous iteration
    
    @JsonProperty("Y0normal") // Variable in normal operation mode Y0
    private double[] Y0normal;
   
    @JsonProperty("X0normal") // variable in normal operation mode X0
    private double[][] X0normal;
    
    
    @JsonProperty("primalObjectiveValue")
    private double primalObjectiveValue;

    
    public AdmmMapperContext() {
    	Lambda = new double[N];
    	Mu = new double[N][N];
    	Y0normal = new double[N];
    	X0normal = new double[N][N];
    	primalObjectiveValue = 0;
  	
    }

       

    @Override
    public void write(DataOutput out) throws IOException {
        Text contextJson = new Text(admmMapperContextToJson(this));
        contextJson.write(out);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        Text contextJson = new Text();
        contextJson.readFields(in);
        setAdmmMapperContext(jsonToAdmmMapperContext(contextJson.toString()));
    }
    
    public void setAdmmMapperContext(AdmmMapperContext context)
    {
    	this.Lambda = context.Lambda;
    	this.Mu = context.Mu;
    	this.Y0normal = context.Y0normal;
    	this.X0normal = context.X0normal;
    	this.primalObjectiveValue = context.primalObjectiveValue;
    }

    public AdmmMapperContext(double[] Lambda, double[][] Mu, double[] Y0normal, double[][] X0normal){
    	this.Lambda = Lambda;
    	this.Mu = Mu;
    	this.Y0normal = Y0normal;
    	this.X0normal = X0normal;

    }
    
    
    @JsonProperty("Lambda")
    public double[] getLambda(){
    	return Lambda; 
    }
    
    @JsonProperty("Mu")
    public double[][] getMu(){
    	return Mu;
    }
    
    @JsonProperty("Y0normal") // Variable in normal operation mode
    public double[] getY0normal(){
    	return Y0normal;
    }
   
    @JsonProperty("X0normal") // variable in normal operation mode
    public double[][] getX0normal(){
    	return X0normal;
    }
    
   

    @JsonProperty("primalObjectiveValue")
    public double getprimalObjectiveValue(){
    	return primalObjectiveValue;
    }
    
    
}