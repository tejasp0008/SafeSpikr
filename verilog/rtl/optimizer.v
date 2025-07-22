module optimizer (
    input clk,
    input rst,
    input spike,
    input signed [7:0] weight_in,
    input signed [7:0] grad_in,
    output reg signed [7:0] weight_out
);

parameter signed [3:0] LEARNING_RATE = 1;

always @(posedge clk or posedge rst) begin
    if (rst)
        weight_out <= 0;
    else if (spike)
        weight_out <= weight_in - LEARNING_RATE * grad_in;
    else
        weight_out <= weight_in;
end

endmodule

