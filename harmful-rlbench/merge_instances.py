import os
import json
import glob
from operator import itemgetter
import argparse

def read_json_from_file(file_path):
    """
    Read and parse JSON content from a file that contains markdown-style JSON blocks.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Extract JSON content between ```json and ``` markers
            json_content = content.split('```json')[1].split('```')[0].strip()
            return json.loads(json_content)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def merge_instances(instance_type='harmful'):
    """
    Merge instances from multiple files based on the instance type.
    
    Args:
        instance_type (str): Type of instances to merge ('harmful' or 'harmless')
    """
    # Get all txt files in the corresponding directory
    if instance_type == 'harmful':
        txt_files = glob.glob(f"./assets/harmful/*.txt")
    else:
        txt_files = glob.glob(f'assets/harmless/*.txt')
    
    # Collect all instances from all files
    all_instances = []
    for txt_file in txt_files:
        instances = read_json_from_file(txt_file)
        all_instances.extend(instances)
        print(f"Successfully loaded {len(instances)} instances from {os.path.basename(txt_file)}")
    
    # Sort instances if it's harmful type
    if instance_type == 'harmful':
        all_instances = sorted(all_instances, key=itemgetter('category-1', 'category-2'))
    
    # Create the output JSON string with proper formatting
    output_json = json.dumps(all_instances, indent=4, ensure_ascii=False)
    
    # Write to a new file
    output_path = f"./assets/{instance_type}_instances.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output_json)
    
    # Print statistics
    print(f"\nMerged {len(txt_files)} files")
    print(f"Total instances: {len(all_instances)}")
    
    # Print category distribution for harmful instances
    if instance_type == 'harmful':
        print("\nCategory distribution:")
        category_dict = {}
        for instance in all_instances:
            cat1 = instance['category-1']
            cat2 = instance['category-2']
            if cat1 not in category_dict:
                category_dict[cat1] = {}
            if cat2 not in category_dict[cat1]:
                category_dict[cat1][cat2] = 0
            category_dict[cat1][cat2] += 1
        
        for cat1 in sorted(category_dict.keys()):
            print(f"\n{cat1}:")
            for cat2 in sorted(category_dict[cat1].keys()):
                print(f"  - {cat2}: {category_dict[cat1][cat2]}")
    
    print(f"\nMerged and sorted file saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge instance files.')
    parser.add_argument('--type', type=str, choices=['harmful', 'harmless'],
                      default='harmful', help='Type of instances to merge (harmful or harmless)')
    
    args = parser.parse_args()
    merge_instances(args.type)