import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.timeseries.eval.TSEvaluation;
import weka.core.Debug;
import weka.core.Instances;
import weka.classifiers.trees.J48;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.attributeSelection.GreedyStepwise;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by cycle on 18.11.16.
 */
public class justJava {
    public static void main(String[] args){
        LinearRegression regression = new LinearRegression();
        try {
            BufferedReader reader = new BufferedReader(
                    new FileReader("/home/cycle/programs/weka-3-8-0/data/iris.arff"));
            Instances data = new Instances(reader);
            reader.close();
            data.setClassIndex(data.numAttributes()-1);
            try {
                AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
                CfsSubsetEval eval = new CfsSubsetEval();
                GreedyStepwise search = new GreedyStepwise();
                search.setSearchBackwards(true);
                J48 base = new J48();
                classifier.setClassifier(base);
                classifier.setEvaluator(eval);
                classifier.setSearch(search);

                try {
                    Evaluation evaluation = new Evaluation(data);
                    evaluation.crossValidateModel(classifier, data, 10, new Debug.Random(1));
                    System.out.println(evaluation.toSummaryString());
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } catch (Exception e){
                e.printStackTrace();
            }
        }catch (IOException e){
            System.out.println("IOException");
        }
    }
}
