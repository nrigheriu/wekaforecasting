import weka.core.Instances;
import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Hashtable;
import java.util.Random;

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
            BitSet b1 = getStartSet(100, 25);
            //System.out.println(includesMoreThan25PercentOfFeatures(b1, 100));
            Random ran = new Random();
            List<String> asd = new ArrayList<String>();
            asd.add(0, "asdas");
            asd.add(1, "asdf");
            asd.add(2, "asdas");
            System.out.println(1e-8*100);
            // evaluate the initial subset
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    protected static BitSet changeBits(int numAttribs, BitSet bitSet){
        Random r = new Random();
        for (int i = 0; i < numAttribs; i++) {
            float chance = r.nextFloat();
            if(chance <= 0.02f){
                if(bitSet.get(i))
                    bitSet.set(i, false);
                else
                    bitSet.set(i, true);
            }
        }
        return bitSet;
    }
    protected static boolean includesMoreThan25PercentOfFeatures(BitSet bitSet, int numAttribs){
        int trueBits = 0;
        for (int i = 0; i < numAttribs; i++) {
            if(bitSet.get(i))
                trueBits++;
        }
        if((float)trueBits/numAttribs > 0.25)
            return true;
        return  false;
    }
    protected static BitSet getStartSet (int numAttribs, int setPercentage){
        BitSet bitSet = new BitSet(numAttribs);
        Random r = new Random();
        for (int i = 0; i < numAttribs; i++) {
            float chance = r.nextFloat();
            if (chance <= 0.25f) {
                bitSet.set(i);
            }
        }
        return bitSet;
    }

}
