module mac (
    input clk,
    input rst,
    input spike_in,                 // 1-bit input
    input signed [7:0] weight,      // 8-bit signed weight
    input signed [15:0] acc_in,     // current membrane potential
    output reg signed [15:0] acc_out // updated potential
);

always @(posedge clk or posedge rst) begin
    if (rst)
        acc_out <= 0;
    else if (spike_in)
        acc_out <= acc_in + weight;
    else
        acc_out <= acc_in;
end

endmodule

