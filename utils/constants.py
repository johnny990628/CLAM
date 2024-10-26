IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
CTRANS_MEAN = [0.485, 0.456, 0.406]
CTRANS_STD = [0.229, 0.224, 0.225]
GIGAPATH_MEAN = [0.485, 0.456, 0.406]
GIGAPATH_STD = [0.229, 0.224, 0.225]


MODEL2CONSTANTS = {
	"resnet50_trunc": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni_v1":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"conch_v1":
	{
		"mean": OPENAI_MEAN,
		"std": OPENAI_STD
	},
	"chief":
	{
		"mean": CTRANS_MEAN,
		"std": CTRANS_STD
	},
	"gigapath":
	{
		"mean": GIGAPATH_MEAN,
		"std": GIGAPATH_STD
	}
}