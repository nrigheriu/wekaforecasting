import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.timeseries.eval.TSEvaluation;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


/**
 * Created by cycle on 14.11.16.
 */
public class example1 {
    public static void main(String[] args){
        LinearRegression regression = new LinearRegression();
        try {
            BufferedReader reader = new BufferedReader(
                    new FileReader("/home/cycle/Downloads/minute_data.arff"));
            Instances data = new Instances(reader);
            reader.close();
            data.setClassIndex(data.numAttributes()-1);
            try {
                //String[] options = new String[1];
                //options[0] = "-S 0";
                //regression.setOptions(options);
                regression.buildClassifier(data);
                TSEvaluation timeEvaluation = new TSEvaluation(data, data.size());
                Evaluation eTest = new Evaluation(data);
                eTest.evaluateModel(regression, data);
                String summary = eTest.toSummaryString();
                System.out.println(summary);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }catch (IOException e){
            System.out.println("IOException");
        }
    }
}
