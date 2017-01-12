package src;

import java.io.*;

import java.text.DecimalFormat;
import java.util.*;
import javax.swing.JPanel;

import weka.attributeSelection.*;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.*;
import weka.classifiers.meta.multisearch.RandomSearch;
import weka.classifiers.timeseries.HoltWinters;
import weka.classifiers.timeseries.TSForecaster;
import weka.classifiers.evaluation.NumericPrediction;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.WekaForecaster;
import weka.classifiers.timeseries.core.TSLagUser;
import weka.classifiers.timeseries.eval.TSEvaluation;
import weka.classifiers.timeseries.eval.graph.JFreeChartDriver;
import weka.classifiers.timeseries.eval.ErrorModule;
import weka.classifiers.timeseries.core.TSLagUser;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Utils.*;
import weka.core.Option;
import weka.filters.Filter;
import weka.filters.supervised.attribute.TSLagMaker;
import weka.attributeSelection.ASSearch;
import weka.core.Option;
public class normalInput {
    public static void main(String[] args) {
        try {
            String pathToWholeData = "/afs/tu-berlin.de/home/n/n_righeriu/irb-ubuntu/workspace/15_min_train+test.arff";
            String pathToHugeData = "/afs/tu-berlin.de/home/n/n_righeriu/irb-ubuntu/workspace/5months_with_weather.arff";
            String pathToTryData = "/afs/tu-berlin.de/home/n/n_righeriu/irb-ubuntu/workspace/practice.arff";

            // load the data
            Instances wholeData  = new Instances(new BufferedReader(new FileReader(pathToWholeData)));
            Instances hugeData = new Instances(new BufferedReader((new FileReader(pathToHugeData))));
            Instances tryData =   new Instances(new BufferedReader((new FileReader(pathToTryData))));

            wholeData.setClassIndex(wholeData.numAttributes()-1);
            hugeData.setClassIndex(hugeData.numAttributes()-1);
            tryData.setClassIndex(tryData.numAttributes()-1);
            //select_Attributes(airlineData);
            //AttributeSelectedClassifier attributeSelectedClassifier = applyMetaClassifier.applyMetaClassifier(airlineData);
            AttributeSelectedClassifier attributeSelectedClassifier2 = new AttributeSelectedClassifier();

            LinearRegression linearRegression = new LinearRegression();
            //linearRegression.setOptions(weka.core.Utils.splitOptions("-S 1"));
            MLPRegressor mlpRegressor = new MLPRegressor();
            //mlpRegressor.setOptions(weka.core.Utils.splitOptions("-N 1"));

            doForecasting.doForecasting(wholeData, applyMetaClassifier.applyMetaClassifier(wholeData));
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
    public Instances select_Attributes(Instances data){
        weka.filters.supervised.attribute.AttributeSelection filter = new weka.filters.supervised.attribute.AttributeSelection();
        WrapperSubsetEval eval = new WrapperSubsetEval();
        ReliefFAttributeEval reliefFAttributeEval = new ReliefFAttributeEval();
        Ranker ranker = new Ranker();
        CfsSubsetEval eval2 = new CfsSubsetEval();
        eval.setClassifier(new LinearRegression());
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        filter.setEvaluator(reliefFAttributeEval);
        filter.setSearch(ranker);
        try {
            filter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, filter);
            System.out.println("new data:" + newData);
            System.out.println(filter);
            return data;
        }catch (Exception e){
            e.printStackTrace();
        }
        return null;
    }
    public void returnSelectedAttributes(){
             /*AttributeSelection attsel  =new AttributeSelection();
            CfsSubsetEval cfsSubsetEval = new CfsSubsetEval();
            GreedyStepwise search = new GreedyStepwise();
            search.setSearchBackwards(true);
            attsel.setEvaluator(cfsSubsetEval);
            attsel.setSearch(search);
            attsel.SelectAttributes(airlineData);
            int[] indices = attsel.selectedAttributes();
            System.out.println("selected attribute :\n" + Utils.arrayToString(indices));*/
    }
}

 /*   int numFolds = 3;

    java.util.Random random = new Random(2);
            for(int i = 0; i < numFolds; ++i) {
        Instances train = tryData.trainCV(numFolds, i, random);
        Classifier copiedClassifier = AbstractClassifier.makeCopy(linearRegression);
        copiedClassifier.buildClassifier(train);
        Instances test = tryData.testCV(numFolds, i);
        }
*/