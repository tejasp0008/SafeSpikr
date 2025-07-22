module neuron (
    input clk,
    input rst,
    input signed [15:0] membrane_potential,
    output reg spike
);

parameter signed [15:0] THRESHOLD = 100;

always @(posedge clk or posedge rst) begin
    if (rst)
        spike <= 0;
    else if (membrane_potential >= THRESHOLD)
        spike <= 1;
    else
        spike <= 0;
end

endmodule

