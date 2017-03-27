import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.ListIterator;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;

public class Question5
{
	
	public static class MeanMapper extends Mapper<Object, Text, Text, DoubleWritable>
	{
		public void map(Object key, Text values, Context context) throws IOException, InterruptedException
		{
			String[] line = StringUtils.split(values.toString(), "^");
			Text BId = new Text(line[2]);
			double rating = Double.parseDouble(line[3]);
			context.write(BId, new DoubleWritable(rating));
		}
	}
	
	public static class MeanReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable>
	{
		 private Map<String, Double> hash1 = new HashMap<String, Double>();
		 
		 public void reduce(Text key, Iterable<DoubleWritable> values, Context context)
		 {
			 int sum = 0, count = 0;
			 for (DoubleWritable value : values)
			 {
				 count += 1;
				 sum += value.get();
			 }
			 hash1.put(key.toString(), (double)sum / (double)count);
		 }
		 
		 protected void cleanup(Reducer<Text, DoubleWritable, Text, DoubleWritable>.Context context) throws IOException, InterruptedException
		 {
			 Map<String, Double> hash2 = Sort(hash1);
			 int count = 0;
			 for (String s : hash2.keySet())
			 {
				 if (count == 10)
				 {
					 break;
				 }
				 count += 1;
				 context.write(new Text(s), new DoubleWritable(hash2.get(s)));
			 }
		 }
		 
		 private static Map<String, Double> Sort(Map<String, Double> map)
		 {
			 List<Map.Entry<String, Double>> list1 = new ArrayList<Entry<String, Double>>(map.entrySet());
			 Collections.sort(list1, new Comparator<Map.Entry<String, Double>>() 
			 {
				 public int compare(Map.Entry<String, Double> x, Map.Entry<String, Double> y)
				 {
					 return(y.getValue()).compareTo(x.getValue());
				 }
			});
			 
			 Map<String, Double> sortMap = new LinkedHashMap<String, Double>();
			 ListIterator<Entry<String, Double>> list2 = list1.listIterator(list1.size());
			 while (list2.hasPrevious())
			 {
				 Map.Entry<String, Double> entry = list2.previous();
				 sortMap.put(entry.getKey(), entry.getValue());
			 }
			 return sortMap;
		 }
	}

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException 
	{
		Configuration conf = new Configuration();
		String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
		if (otherArgs.length != 2)
		{
			System.err.println("Usage : Question5 <input file path> <output folder path>");
		}
		
		Job job = Job.getInstance(conf, "Question5");
		job.setJarByClass(Question5.class);
		job.setMapperClass(MeanMapper.class);
		job.setReducerClass(MeanReducer.class);
		
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(DoubleWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
		FileOutputFormat.setOutputPath(job, new Path(otherArgs[1]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}

}
