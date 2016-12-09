import weka.core.Instances;
import java.io.*;
import weka.classifiers.functions.LinearRegression;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Utils;

/**
 * Created by cycle on 10.11.16.
 */
public class anotherExample {

    public static void main(String[] args) {
        try {
            Instances airlineData = DataSource.read("/home/cycle/workspace/airline.arff");
            WekaForecaster forecaster = new WekaForecaster();
            forecaster.setOptions(Utils.
                    splitOptions("-F passenger_numbers -G Date -quarter -month -W \"weka.classifiers.meta.AttributeSelectedClassifier -W weka.classifiers.functions.LinearRegression\" -prime 12"));
            forecaster.buildForecaster(airlineData, System.out);
            System.out.println(forecaster);
        } catch (Exception e) {
            e.printStackTrace();
        }

        // setting class attribute

    }
}
