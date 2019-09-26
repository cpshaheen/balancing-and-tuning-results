import weka.core.converters.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.FastVector;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.core.DenseInstance;

import weka.filters.unsupervised.attribute.Remove;
import weka.core.Attribute;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.core.AttributeStats;
import weka.classifiers.trees.RandomForest;
import weka.experiment.Stats;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import java.util.Random;

import java.io.*;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays.*;
import java.util.*;
import java.lang.*;
import java.lang.Object.*;

public class ModelRunner{
  public static void main(String args[]){
    /*usage:
    please have the following directory ordering:
      ./outtermostdir/sampledataprojectdir/datasets/projname_samplingtechnique, camel_test.csv, camel_UNSAMPLED.csv
    call in terminal in the following format:
      "java ModelRunner ./outtermostdir/*"
    */
    ArrayList<String> dirNames = new ArrayList<String>(Arrays.asList(args));
    //for each directory

    ArrayList<String> pF = new ArrayList<String>(Arrays.asList(Arrays.copyOfRange(args,0,args.length)));
    ArrayList<String> pFmod = new ArrayList<String>();

    for(String s: pF){
      pFmod.add(s.substring(s.indexOf("/")+1,s.lastIndexOf("_")));
    }

    System.out.println(pFmod);

    ArrayList<String> c = new ArrayList<String>();

    c.add("RF");

    Instances res = csvCreator(pFmod,c);

    for(String dir: dirNames){
      retrieveCsvs(dir,res);
    }

    try{

        //here we need to replace this print with something like
        //csvSaver(dataUSampled, somefilename)
        String resultsFileName = args[0].substring(0,args[0].indexOf("_")) + "_Evaluation_Results.csv";
        saveCsv(resultsFileName,res);
        //System.out.println(dataUSampled);

        }
         catch(Exception e)
         {
           e.printStackTrace();
         }
  }

  public static void retrieveCsvs(String dirName, Instances results){
    //this funcitons role is to go through the sampled data folder of a particular
    //project (ie; ant,forrest,xalan -etc) to retrieve the test data, and
    //training data (ie; oversampled, undersampled, not sampled)
    //this will make it easier to use a particular sampled data and evaluate
    //with the test data

    //**TODO**create csv file to append information to following each model run
    System.out.println("\n\n\t"+dirName);

    //this is a result of the WekaCsv.java programs directory saving
    //not a bug just a minor inconvienience (look at a sampled datasets
    //folder to understand the odd directory format)
    dirName = dirName + "/datasets";

    ArrayList<File> samplingDirs = new ArrayList<File>();
    File dir = new File(dirName);
    File nonSampledData = null;
    File testData = null;
    Boolean trainAvail = false;

    //ignore .DS_Store only ackn
    if(!dir.exists()){
      System.out.println(dirName);
      System.out.println("FILE DNE");
      return;
    }
    for (File filelvl0 : dir.listFiles()) {
        if (filelvl0.isDirectory()){
          //add to list of directories to later then run a model runner on all
          //files in the directories
            samplingDirs.add(filelvl0);
            //System.out.println(filelvl0.getName());
        }
        else if(filelvl0.getName()!=".DS_Store"){
          //if file isnt DSSTORE it is either ends in train.csv or test.csv
            if(filelvl0.getName().contains("test.csv")){
              testData = filelvl0;
            }
            if(filelvl0.getName().contains("train.csv")){
              nonSampledData = filelvl0;
            }
        }
    }

    if(testData==null){
      System.out.println("ERROR: MISSING \"*test.csv\"");
      System.out.println("Directory: "+dirName);
      System.exit(-1);
    }
    else{
      System.out.println("Test Data:\t" + testData);
    }

    if(nonSampledData!=null){
      trainAvail = true;
      System.out.println("Non-Sampled Training Data:\t" + nonSampledData.getName());
      rfModelRunner(nonSampledData,testData, results, "UNSAMPLED");
    }

    for(File sampleDir : samplingDirs){
      if(sampleDir.getName().contains("OVERSAMPLED")){
        trainAvail = true;
        System.out.println("OverSampled Data: "+ sampleDir);
        for(File actualDir : sampleDir.listFiles()){
          if(!actualDir.getName().contains(".DS_Store")){
            System.out.println(actualDir);
            for(File filelvl1 : actualDir.listFiles()){
              rfModelRunner(filelvl1,testData,results,"OVERSAMPLED");
            }
          }
        }
      }
      else if(sampleDir.getName().contains("UNDERSAMPLED")){
        trainAvail = true;
        System.out.println("UnderSampled Data: "+ sampleDir);
        for(File actualDir : sampleDir.listFiles()){
          if(!actualDir.getName().contains(".DS_Store")){
            System.out.println(actualDir);
            for(File filelvl1 : actualDir.listFiles()){
              rfModelRunner(filelvl1,testData,results,"UNDERSAMPLED");
            }
          }
        }
      }
      else if(sampleDir.getName().contains("SMOTE")){
        trainAvail = true;
        System.out.println("SMOTE Data: "+sampleDir);
        for(File actualDir : sampleDir.listFiles()){
          if(!actualDir.getName().contains(".DS_Store")){
            System.out.println(actualDir);
            for(File filelvl1 : actualDir.listFiles()){
              rfModelRunner(filelvl1,testData,results,"SMOTE");
            }
          }
        }
      }
    }
    if(trainAvail=false){
      System.out.println("ERROR: MISSING ANY TRAINING FILES FOR \""+dirName+"\"");
      System.exit(-1);
    }



    //then run the train.csv file(add its results to the csv results file)
    //then for everydir run each file(add its results to csv results file)
    //***make func that takes .csv file and test.csv file and adds results
    //   to output csv file for each model***
  }

  public static void rfModelRunner(File traindata, File test, Instances res, String treatment){
   RandomForest rf = new RandomForest();         // new instance of tree
   Instances idata = null;
   try{
     idata = masterLoader(traindata);
     Evaluation eval = new Evaluation(idata);
     eval.crossValidateModel(rf, idata, 11, new Random(1));
     //System.out.println(eval.toSummaryString("\nResults\n======\n", false));
     String dataset = traindata.getName();
     String sampleID = dataset.substring(dataset.lastIndexOf("_"),dataset.lastIndexOf(".csv"));

     DenseInstance di = new DenseInstance(res.numAttributes());
     di.setDataset(res);
     di.setValue(0, dataset);
     di.setValue(1, treatment);
     di.setValue(2, "RANDOM FOREST");
     di.setValue(3, sampleID);
     di.setValue(4, eval.truePositiveRate(idata.numClasses()-1));
     di.setValue(5, eval.falsePositiveRate(idata.numClasses()-1));
     di.setValue(6, eval.trueNegativeRate(idata.numClasses()-1));
     di.setValue(7, eval.falseNegativeRate(idata.numClasses()-1));
     di.setValue(8, eval.precision(idata.numClasses()-1));
     di.setValue(9, eval.weightedAreaUnderROC());
     di.setValue(10, eval.kappa());
     System.out.println(di);
     res.add(di);
   }
   catch(Exception e){
       e.printStackTrace();
       System.exit(-1);
   }
 }

 //this function will return an instance to use for an evaluator
 //will be used in each of desired models trainers/evaluators
  public static Instances masterLoader(File data){
    RandomForest rf = new RandomForest();         // new instance of tree
    CSVLoader loader = new CSVLoader();
    Instances idata = null;
    try{
      loader.setSource(data);
    }
    catch(IOException e){
      e.printStackTrace();
      System.exit(-1);
    }
    try{
      idata = loader.getDataSet();
      idata.setClassIndex(idata.numAttributes()-1);
      return idata;
    }
    catch(Exception e){
      e.printStackTrace();
      System.exit(-1);
      return null;
     }
  }

  public static Instances csvCreator(ArrayList<String> projNames, ArrayList<String> classifiers){

          File results = new File("");

          ArrayList<Attribute> atbs = new ArrayList<Attribute>();

          Attribute dataset = new Attribute("Dataset", (FastVector) null);
          ArrayList<String> treatments = new ArrayList<String>();
          treatments.add("NS");
          treatments.add("OS");
          treatments.add("US");
          treatments.add("S");
          treatments.add("ST");
          Attribute treatment = new Attribute("Treatment", (FastVector)null);
          Attribute clsf = new Attribute("Classifier", (FastVector)null);
          Attribute sampleID = new Attribute("Sample ID", (FastVector)null);
          Attribute tp = new Attribute("True Positive");
          Attribute fp = new Attribute("False Positive");
          Attribute tn = new Attribute("True Negative");
          Attribute fn = new Attribute("False Negative");
          Attribute precision = new Attribute("Precision");
          Attribute auc = new Attribute("AUC");
          Attribute kappa = new Attribute("Kappa");

          atbs.add(dataset);
          atbs.add(treatment);
          atbs.add(clsf);
          atbs.add(sampleID);
          atbs.add(tp);
          atbs.add(fp);
          atbs.add(tn);
          atbs.add(fn);
          atbs.add(precision);
          atbs.add(auc);
          atbs.add(kappa);
          Instances data = new Instances("test", atbs, 11);

          NumericToNominal convert= new NumericToNominal();
          convert.setAttributeIndices("1,2,3");
          try{
          convert.setInputFormat(data);
          data=Filter.useFilter(data, convert);
          }
          catch(Exception e){
            e.printStackTrace();
          }


               return data;
  }

  public static void saveCsv(String path, Instances data) throws IOException{
    System.out.println("\nSaving to file " + path + "...");
    CSVSaver saver = new CSVSaver();
    saver.setInstances(data);
    saver.setFile(new File(path));
    saver.writeBatch();
  }

  //takes a list of csv files and uses the weka api CSVLoader to create and
  //return weka Instances to be later used by the specified data balancing
  //techinque through the use of the weka Filter
  public static ArrayList<Instances> loader(ArrayList<String> csvFileNames){
    ArrayList<Instances> csvData = new ArrayList<>();

    for(String csvFileName : csvFileNames){
      if(!csvFileName.endsWith(".csv")){
        System.err.println("Specified file: " + csvFileName + " does not have"+
                            " the .csv file extention\nPLEASE USE .CSV FILES");
        System.exit(-1);
      }
      //here we open the file and use the weka api's CSVLoader
      File file = new File(csvFileName);
      CSVLoader loader = new CSVLoader();

      try{
        loader.setSource(file);
      }
      catch(IOException e){
        e.printStackTrace();
        System.exit(-1);
      }

      try{
        Instances data = loader.getDataSet();
        //data.setClassIndex(data.numAttributes()-1);
        csvData.add(data);
      }
      catch(Exception e){
          e.printStackTrace();
          System.exit(-1);
      }
    }
    return csvData;
  }
}
