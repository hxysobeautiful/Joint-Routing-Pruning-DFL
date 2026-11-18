from pathlib import Path

from client import *
from cv1 import *
import models, datasets
def zero_out_rows_with_probability(matrix, p):
    # num_rows = matrix.size(0)b = {Tensor: ()} tensor(1., device='cuda:0', dtype=torch.float64)
    num_tensors = len(matrix)  # 3990
    p = p.item()

    # 生成一个随机掩码（mask），决定哪些张量会被置零
    mask = torch.rand(num_tensors) < p  # True=保留，False=置零

    # 遍历矩阵，对未选中的张量置零
    for i in range(num_tensors):
        if not mask[i]:  # 如果 mask[i] == False，则置零
            matrix[i] = torch.zeros_like(matrix[i])  # 全部填充 0

    return matrix,mask

class ClientModelProcessor:
    def __init__(self, global_parameters, batch_size):
        """
        初始化客户端模型处理器

        :param models_last_round: 包含多个客户端模型的列表，每个元素为 (样本数量, 模型参数字典)
        :param batch_size: 每个数据包的大小
        """
        self.batch_size = batch_size
        self.client_packets = {}
        self.new_models={}
        self.global_parameters=global_parameters

    def get_model_params_with_metadata(self, model):
        params_with_metadata = OrderedDict()  # 使用 OrderedDict 保证顺序
        for name, param in model.named_parameters():
            params_with_metadata[name] = param.clone().detach()  # 直接存储 tensor

        return params_with_metadata

    def process_other_client_models(self, reordered_models):
        client_packets = []
        all_params_tensor = reordered_models

        # 计算完整的数据包数量（直接整除，向下取整）
        num_packets = len(all_params_tensor) // self.batch_size

        # 只处理完整的批次
        for i in range(num_packets):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            packet = all_params_tensor[start_idx:end_idx]
            client_packets.append(packet)

        # 可选：打印被舍弃的尾部数据量
        remaining = len(all_params_tensor) % self.batch_size
        # if remaining > 0:
        #     print(f"舍弃了最后 {remaining} 个参数（不足一个完整批次）")

        self.client_packets = client_packets
        return client_packets

    def process_client_models(self,reordered_models):
        client_packets = []
            # 遍历客户端模型中的每个参数（权重、偏置等）

            # 将所有参数连接成一个大的tensor
        all_params_tensor = reordered_models

            # 计算数据包的数量
        num_packets = len(all_params_tensor) // self.batch_size

        if len(all_params_tensor) % self.batch_size != 0:
                num_packets += 1

            # 将参数划分成多个数据包
        for i in range(num_packets):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(all_params_tensor))
                packet = all_params_tensor[start_idx:end_idx]
                client_packets.append(packet)

            # 保存客户端的所有数据包
        self.client_packets = client_packets


        return self.client_packets

    def centralized_PER_replace_with_zero(self,global_parameters,reordered_models):#用于计算数据包中的通信错误率的影响
        device = torch.device("cuda:0")
        allow_pickle = True

        route_dir = Path(os.environ.get(
            "FL_ROUTE_DIR",
            str(Path(__file__).resolve().parents[2] / "seed42" / "proposed")
        ))
        one_hop_error = np.loadtxt(route_dir / "one_hop_error", delimiter=',')


        path = np.load(route_dir / "pathbest.npy", allow_pickle=True)

        path = torch.tensor(path).to(device)

        #
        r_m = np.loadtxt(route_dir / "r_m.txt", delimiter=',')

        r_m = torch.tensor(r_m).to(device)
        client_model_processor = ClientModelProcessor(global_parameters, batch_size=500)

        for i in range(len(reordered_models)):

            params = [None for _ in range(50000)]
            counts = [None for _ in range(50000)]
            original_reordered_models= copy.deepcopy(reordered_models)# ✅ 每个 param[p] 都配一个 count[p]

            for k ,v in enumerate(original_reordered_models):


               # #pruning
                rm = r_m[k][i]



                num_to_keep = int(len(v) * rm)

                reordered_models_parmes =v[:num_to_keep]
                if i==k:
                    client_packets_pruning = client_model_processor.process_client_models(reordered_models_parmes)
                else:
                    client_packets_pruning = client_model_processor.process_other_client_models(reordered_models_parmes)

                a=1

                path_nodes = len(path[k][i])

                for j in range(path_nodes - 1):
                    a = a * one_hop_error[path[k][i, j], path[k][i, j + 1]]

#裁剪后的模型划分为数据包并进行了上传错误率处理
                value_pruning, mask = zero_out_rows_with_probability(client_packets_pruning, p=a)

                for p, packet in enumerate(value_pruning):
                    if counts[p] is None:
                        counts[p] = torch.zeros_like(packet)

                    if mask[p] == True:
                        counts[p] +=torch.ones_like(packet) # 累加成功接收的数量

                    if params[p] is None:
                        params[p] = packet.clone()  # ✅ 拷贝一份，防止引用原来的 tensor
                    else:
                        params[p] += packet

            # ✅ 聚合完成后，对每个参数除以成功接收次数，防止除以 0
            for p in range(len(params)):
                if params[p] is not None:
                    safe_count = counts[p] + 1e-8  # 避免除以 0
                    params[p] = params[p] / safe_count

            params = [x for x in params if x is not None]
            self.new_models[i] = copy.deepcopy(params)

        return self.new_models

    def restore_model_from_flat(self,model_template, flat_param, sorted_indices, shapes):
        # 逆映射：sorted_indices => 原始位置
        inverse_idx = torch.argsort(sorted_indices)
        # 检查并清理数据
        all_params = []
        for k, v in enumerate(flat_param):
            # 将每个参数矩阵展平成一维向量
            all_params.append(v.view(-1))

            # 将所有参数连接成一个大的tensor
        all_params_tensor = torch.cat(all_params)
        all_params_tensor = torch.tensor( all_params_tensor, device=inverse_idx.device, dtype=torch.float32)

        original_order_param =  all_params_tensor[inverse_idx]

        # 拆成参数块
        pointer = 0
        new_state_dict = OrderedDict()


        for (name, param), shape in zip(self.global_parameters.items(), shapes):
            length = torch.tensor(shape).prod().item()
            param_tensor = original_order_param[pointer:pointer + length].view(shape)
            new_state_dict[name] = param_tensor
            pointer += length

        # 加载到模型中
        values_list = list(new_state_dict.values())
        # model_template.load_state_dict(new_state_dict)
        return  values_list

    def restore_model(self):
        """
        恢复模型参数，填充为0的部分将被恢复为原始大小
        """
        restored_models = {}


        for client_id, packets in self.new_models.items():
            # 连接数据包，恢复为一个大的张量
            restored_tensor = torch.cat(packets)
            # 将恢复后的张量重塑回模型参数的原始形状
            model_params = {}
            idx = 0
            for k, v in self.global_parameters.items():
                # 计算该参数的尺寸
                param_size = v.numel()
                # 截取对应的参数并赋值回模型
                restored_tensor_k = restored_tensor[idx:idx + param_size]
                model_params[k] = restored_tensor_k.view(v.shape)
                idx += param_size

            # 将恢复后的模型存储
            restored_models[client_id] = model_params

        return restored_models

    def get_client_packets(self):
        """
        获取所有客户端的划分数据包
        """
        return self.client_packets
    def get_new_model_packets(self):
        """
        获取所有客户端的划分数据包
        """
        return self.new_models

