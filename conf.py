
##配置文件
conf = {

	#选择模型
	"model_name" : "mlp",

	#全局epoch
	"global_epochs" : 1,

	#本地epoch
	"local_epochs" : 3,


	"batch_size" : 64,

    #学习速率
	"lr" : 0.001,

	"momentum" : 0.0001,

	#分类
	"num_class": 2,

    #模型聚合权值
	"is_init_avg": True,

    #本地验证集划分比例
	"split_ratio": 0.3,

    #标签列名
	"label_column": "label",

    #测试数据
	"test_dataset": "./data/adult/adult_test.csv",

    #训练数据
	"train_dataset" : {
        "alice": "./data/adult/adult_part_0.csv",
        "bob": "./data/adult/adult_part_1.csv",
        "lace": "./data/adult/adult_part_2.csv",
        "laodou":"./data/adult/adult_part_3.csv"
	},

    #模型保存目录
	"model_dir":"./save_model/",

    #模型文件名
	"model_file":"model.pth"
}