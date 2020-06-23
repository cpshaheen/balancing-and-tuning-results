import weka.core.converters.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Remove;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.experiment.Stats;

import java.io.*;
import java.util.Scanner;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.*;
import java.lang.*;
import java.lang.Object.*;

public class DataBalancer{
  public File test;
  public static void main(String[] args){
    Boolean usample = false;
    Boolean osample = false;
    Boolean smotesample = false;
    if(args[0]==null){
      System.err.println("USAGE: [-uos] examplefile1.csv examplefile2.csv ...");
      System.exit(-1);
    }
    if(args[0].charAt(0)!='-'){
      System.err.println("USAGE: [-uos] examplefile1.csv examplefile2.csv ...");
      System.exit(-1);
    }
    if(args[0].length()>4 || args[0].length()<1){
      System.err.println("USAGE: [-uos] examplefile1.csv examplefile2.csv ...");
      System.exit(-1);
    }
    else{
      if(args[0].contains("u")){
          System.out.println("usample");
          usample = true;
      }
      if(args[0].contains("o")){
          System.out.println("osample");
          osample = true;
      }
      if(args[0].contains("s")){
          System.out.println("smotesample");
          smotesample = true;
      }
      if((usample || osample || smotesample)==false){
        System.err.println("FLAG MUST BE SPECIFIED WITH" +
                            "AT LEAST ONE OF THE FOLLOWING OPTIONS:\n" +
                            "\t-o Oversampling\n" +
                            "\t-u Undersampling\n" +
                            "\t-s Smote");
        System.exit(-1);
        }
      }

    ArrayList<String> listOfNames = new ArrayList<String>(Arrays.asList(Arrays.copyOfRange(args,1,args.length)));

    //we call the loader to create a list of the loaded data wrapped in
    //the weka api Instances class
    ArrayList<Instances> loadedData = loader(listOfNames);
    ArrayList<Instances> listOfProcData = new ArrayList<Instances>();
    String[] options = new String[2];
    options[0] = "-R";
    options[1] = "1,26,24,23";

    for(Instances i : loadedData){
      Remove rmvFilter = new Remove();
      try{
        rmvFilter.setOptions(options);
      rmvFilter.setInputFormat(i);
      //rmvFilter.setAttributeIndicesArray(columnsToRemove);
      Instances procData = Filter.useFilter(i, rmvFilter);
      listOfProcData.add(procData);
      }
      catch(Exception e){
        e.printStackTrace();
      }
    }

    ArrayList<Instances> testData = new ArrayList<Instances>();
    ArrayList<Instances> trainData = new ArrayList<Instances>();

    for(Instances i : listOfProcData){
      int trainSize = (int) Math.round(i.numInstances() * 66/100);
      int testSize = i.numInstances() - trainSize;
      Instances train = new Instances(i, 0, trainSize);
      trainData.add(train);
      Instances test = new Instances(i, trainSize, testSize);
      testData.add(test);
    }

    ArrayList<String> listOfDirs = topLevelDirMaker(listOfNames);

    for(int i = 0; i<testData.size(); i++){
      String testName;
      String trainName;
      testName = listOfNames.get(i).substring(0, listOfNames.get(i).lastIndexOf('.'));
      trainName = listOfDirs.get(i) + "/" + testName + "_" + "UNSAMPLED_train.csv";
      testName = listOfDirs.get(i) + "/" + testName + "_" + "test.csv";

      try{
        saveCsv(testName, testData.get(i));
        saveCsv(trainName, trainData.get(i));
      }
      catch(Exception e){
        e.printStackTrace();
      }
    }


    //if oversampling; feed list of loaded data, csv names and their directories
    // to oversampler function
    if(osample){overSampleCaller(trainData,listOfNames,listOfDirs);}
    //if undersampling; feed list of loaded data, csv names and their directories
    // to undersampler function
    if(usample){underSampleCaller(trainData,listOfNames,listOfDirs);}
    //if smotesampling; feed list of loaded data, csv names and their directories
    // to smotesampler function
    if(smotesample){smoteCaller(trainData, listOfNames, listOfDirs);}

  }

  public static ArrayList<String> topLevelDirMaker(ArrayList<String> listOfNames){
    //create directories with the following naming convention
    // csvFileName_todaysDate
    ArrayList<String> listOfDirs = new ArrayList<String>();
    for(String name : listOfNames){
      String newName;
      newName = name.substring(0, name.lastIndexOf('.'));
      newName = newName + "_" + java.time.LocalDate.now();
      File dir = new File(newName);
      dir.mkdir();
      listOfDirs.add(newName);
    }
    return listOfDirs;
  }

  //undersampling function will take the list of Instances and
  //for each one make 100 undersampled data subsets
  public static void smoteCaller(ArrayList<Instances> listOfData, ArrayList<String> listOfNames, ArrayList<String> listOfDirs){
    //  given dir csvFileName_todaysDate
    //  ->csvFileName_typeOfSampler_numberofSamples

    // create lists of each csvFileName's sampler dirs
    for(int i = 0; i< listOfData.size(); i++){
      Instances data;
      String fileDir;
      String fileName;
      data = listOfData.get(i);
      fileName = listOfNames.get(i);
      fileName = fileName.substring(0, fileName.lastIndexOf('.'));
      fileDir = fileName + "_SMOTE_" + "100";
      fileDir = listOfDirs.get(i) + "/" + fileDir;
      File unsDir = new File(fileDir);
      unsDir.mkdir();
      smoter(data, fileDir, fileName, 100); //function takes the instances wrapped data and an
                                    //integer amount of samples to create
    }
  }

  //undersampling function will take the list of Instances and
  //for each one make 100 undersampled data subsets
  public static void underSampleCaller(ArrayList<Instances> listOfData, ArrayList<String> listOfNames, ArrayList<String> listOfDirs){
    //  given dir csvFileName_todaysDate
    //  ->csvFileName_typeOfSampler_numberofSamples

    // create lists of each csvFileName's sampler dirs
    for(int i = 0; i< listOfData.size(); i++){
      Instances data;
      String fileDir;
      String fileName;
      data = listOfData.get(i);
      fileName = listOfNames.get(i);
      fileName = fileName.substring(0, fileName.lastIndexOf('.'));
      fileDir = fileName + "_UNDERSAMPLED_" + "100";
      fileDir = listOfDirs.get(i) + "/" + fileDir;
      File unsDir = new File(fileDir);
      unsDir.mkdir();
      underSample(data, fileDir, fileName, 100); //function takes the instances wrapped data and an
                                    //integer amount of samples to create
    }
  }

  //oversampling function will take the list of Instances and
  //for each one make 100 oversampled data subsets
  public static void overSampleCaller(ArrayList<Instances> listOfData, ArrayList<String> listOfNames, ArrayList<String> listOfDirs){
    //  given dir csvFileName_todaysDate
    //  ->csvFileName_typeOfSampler_numberofSamples

    // create lists of each csvFileName's sampler dirs
    for(int i = 0; i< listOfData.size(); i++){
      Instances data;
      String fileDir;
      String fileName;
      data = listOfData.get(i);
      fileName = listOfNames.get(i);
      fileName = fileName.substring(0, fileName.lastIndexOf('.'));
      fileDir = fileName + "_OVERSAMPLED_" + "100";
      fileDir = listOfDirs.get(i) + "/" + fileDir;
      File ovsDir = new File(fileDir);
      ovsDir.mkdir();
      overSample(data, fileDir, fileName, 100); //function takes the instances wrapped data and an
                                    //integer amount of samples to create
    }
  }

  public static void smoter(Instances data, String fileDir, String fileName, int numSamples){
    //first we must calc the % of smote instances to create (which will be
    // equal to the maj class divided by the min class * 100)
    int majClass;
    int minClass;
    AttributeStats aStats = data.attributeStats(data.numAttributes()-1);
    int nC[] = aStats.nominalCounts;
    if(nC[0]>nC[1]){
      majClass = nC[0];
      minClass = nC[1];
    }
    else{
      majClass = nC[1];
      minClass = nC[0];
    }
    double smotePercentage = ((double)(majClass-minClass)/(double)minClass)*100;

    for(int i = 0; i<numSamples; i++){
      String csvFileName = fileName + "_SMOTE_" + String.valueOf(i) + ".csv";
      String path = fileDir + "/" + csvFileName;
      data.setClassIndex(data.numAttributes()-1);
      SMOTE smote = new SMOTE();
      smote.setRandomSeed((int)System.currentTimeMillis());
      try{
        smote.setInputFormat(data);
        smote.setPercentage((int)smotePercentage);

          try{
              Instances dataUSampled = Filter.useFilter(data, smote);
              //here we need to replace this print with something like
              //csvSaver(dataUSampled, somefilename)
              saveCsv(path, dataUSampled);
              //System.out.println(dataUSampled);
              }
               catch(Exception e)
               {
                 e.printStackTrace();
               }
      }
      catch(Exception e)
      {
        e.printStackTrace();
      }
    }
  }

  public static void underSample(Instances data, String fileDir, String fileName, int numSamples){
    for(int i = 0; i<numSamples; i++){
      String csvFileName = fileName + "_UNDERSAMPLED_" + String.valueOf(i) + ".csv";
      String path = fileDir + "/" + csvFileName;
      data.setClassIndex(data.numAttributes()-1);
      SpreadSubsample s = new SpreadSubsample();
      s.setRandomSeed((int)System.currentTimeMillis());
      try{
        s.setInputFormat(data);
        s.setDistributionSpread(1.0);
          try{
              Instances dataUSampled = Filter.useFilter(data, s);
              //here we need to replace this print with something like
              //csvSaver(dataUSampled, somefilename)
              saveCsv(path, dataUSampled);
              //System.out.println(dataUSampled);
              }
               catch(Exception e)
               {
                 e.printStackTrace();
               }
      }
      catch(Exception e)
      {
        e.printStackTrace();
      }
    }
  }

  public static void overSample(Instances data, String fileDir, String fileName, int numSamples){
    //first we must calc the % of the maj class
    int numMajClass = 0;
    AttributeStats aStats = data.attributeStats(data.numAttributes()-1);
    int nC[] = aStats.nominalCounts;
    if(nC[0]>nC[1]){
      numMajClass = nC[0];
    }
    else{
      numMajClass = nC[1];
    }
    double ssPercent = ((double)numMajClass/(double)data.numInstances())*100;

    for(int i = 0; i<numSamples; i++){
      String csvFileName = fileName + "_OVERSAMPLED_" + String.valueOf(i) + ".csv";
      String path = fileDir + "/" + csvFileName;
      data.setClassIndex(data.numAttributes()-1);
      Resample rs = new Resample();
      rs.setRandomSeed((int)System.currentTimeMillis());
      try{
        rs.setInputFormat(data);
        rs.setNoReplacement(false);
        rs.setBiasToUniformClass(1.0);
        rs.setSampleSizePercent(ssPercent*2);
          try{
              Instances dataUSampled = Filter.useFilter(data, rs);

              //here we need to replace this print with something like
              //csvSaver(dataUSampled, somefilename)
              saveCsv(path, dataUSampled);
              //System.out.println(dataUSampled);

              }
               catch(Exception e)
               {
                 e.printStackTrace();
               }
      }

      catch(Exception e)
      {
        e.printStackTrace();
      }
    }
  }

    public static void saveCsv(String path, Instances data) throws IOException{
	    System.out.println("\nSaving to file " + path + "...");
	    CSVSaver saver = new CSVSaver();
	    saver.setInstances(data);
	    saver.setFile(new File(path));
	    saver.writeBatch();
    }


  //undersampling function will take the list of Instances and
  //for each one make 100 undersampled data subsets

  //smotesampling function will take the list of Instances and
  //for each one make 100 smotesampled data subsets

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
